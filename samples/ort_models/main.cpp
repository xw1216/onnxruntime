// ==============================================
// Multi-model image inference demo (ResNet18 + YOLO)
// - ResNet18: load image, resize to 224x224, normalize with ImageNet mean/std, print top-5 labels
// - YOLO: load image, resize to 640x640 (simple stretch), parse first output tensor, print detection boxes
// Dependencies: third_party/stb_image.h, stb_image_resize2.h, labels/imagenet_classes.txt, labels/coco_classes.txt
// Command line:
//   --model <resnet|yolo|embed|all>
//   --image <image_path> (used by resnet / yolo)
//   --labels <imagenet_label_file> (default ./third_party/imagenet_classes.txt)
//   --yolo-thresh <float> (default 0.25)
// ==============================================

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cctype>
#include <onnxruntime_cxx_api.h>
#include <unistd.h> // readlink to get executable path

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb_image_resize2.h"

namespace fs = std::filesystem;

static void Usage(){
  std::cout << "Usage: ort_models --model <resnet|yolo|all> --image <path>\n";
  std::cout << "       [--labels imagenet_file] [--yolo-labels coco_file]\n";
  std::cout << "       [--yolo-thresh v] [--yolo-nms v]\n";
  std::cout << "       [--warmup N] [--repeat M]\n";
  std::cout << "Notes: by default looks for models/ and labels/ relative to executable directory.\n";
  std::cout << "       --yolo-thresh confidence threshold (default 0.25)  --yolo-nms NMS IoU threshold (default 0.45)\n";
  std::cout << "       Timing always enabled; --warmup warmup runs (default 0), --repeat timed runs (default 1)\n";
}
// Get directory of current executable (Linux/OHOS uses /proc/self/exe)
static std::string GetExecutableDir(){
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf)-1);
  if(n > 0){
    buf[n] = '\0';
    std::string p(buf);
    auto pos = p.find_last_of('/');
    if(pos != std::string::npos) return p.substr(0,pos);
  }
  return std::string(".");
}
static void ToLower(std::string& s){ for(auto& c:s) c=(char)std::tolower((unsigned char)c); }

// Read all non-empty lines
static std::vector<std::string> ReadLines(const std::string& file){
  std::vector<std::string> lines; std::ifstream fin(file); std::string line; while(std::getline(fin,line)) if(!line.empty()) lines.push_back(line); return lines; }

// Simple resize + convert to float (CHW)
static bool LoadImageToCHWFloat(const std::string& path, int target_w, int target_h, std::vector<float>& out, bool normalize, int* orig_w, int* orig_h){
  int w,h,c; unsigned char* data = stbi_load(path.c_str(), &w,&h,&c, 3); if(!data) return false; c=3; if(orig_w) *orig_w=w; if(orig_h) *orig_h=h;
  std::vector<unsigned char> resized((size_t)target_w*target_h*c);
  // Use stb resize, input stride = w*c, output stride = target_w*c
  if(w!=target_w || h!=target_h){
    stbir_resize_uint8_linear(data, w, h, w*c, resized.data(), target_w, target_h, target_w*c, STBIR_RGB);
  } else {
    std::memcpy(resized.data(), data, (size_t)w*h*c);
  }
  stbi_image_free(data);
  out.assign((size_t)c*target_w*target_h, 0.f);
  // Convert to CHW layout and scale to [0,1]
  for(int yy=0; yy<target_h; ++yy){
    for(int xx=0; xx<target_w; ++xx){
      size_t src_i = (size_t)(yy*target_w + xx)*c;
      for(int ch=0; ch<c; ++ch){
        float v = resized[src_i+ch]/255.0f;
        out[(size_t)ch*target_w*target_h + yy*target_w + xx] = v;
      }
    }
  }
  if(normalize){
    const float mean[3] = {0.485f,0.456f,0.406f};
    const float stdv[3] = {0.229f,0.224f,0.225f};
    for(int ch=0; ch<3; ++ch){
      float* plane = out.data() + (size_t)ch*target_w*target_h;
      for(int i=0;i<target_w*target_h;++i){ plane[i] = (plane[i]-mean[ch])/stdv[ch]; }
    }
  }
  return true;
}

// softmax + topk
struct Ranked { int idx; float val; };
static void TopK(const float* logits, size_t n, int k, std::vector<Ranked>& out){
  out.clear(); out.reserve(n);
    // Numerically stable: subtract max
  float m = -1e30f; for(size_t i=0;i<n;++i) if(logits[i]>m) m=logits[i];
  std::vector<float> probs(n); float sum=0.f; for(size_t i=0;i<n;++i){ probs[i]=std::exp(logits[i]-m); sum += probs[i]; }
  for(size_t i=0;i<n;++i){ probs[i]/=sum; }
  for(size_t i=0;i<n;++i) out.push_back({(int)i, probs[i]});
  std::partial_sort(out.begin(), out.begin()+std::min<int>(k,out.size()), out.end(), [](const Ranked&a,const Ranked&b){return a.val>b.val;});
  if((int)out.size()>k) out.resize(k);
}

// YOLO parsing: supports two formats
// 1) [1,N,A], A>=6 interpreted as (cx,cy,w,h,obj,cls_probs...)
// 2) [1,N,6] interpreted as (x1,y1,x2,y2,score,class_id) -- current user model: shape=[1,300,6]
struct YoloDet { float score; float x1,y1,x2,y2; int cls; float obj; };
static std::vector<YoloDet> ParseYolo(const float* data, int N, int A, float thresh){
  std::vector<YoloDet> dets; if(A<6) return dets;
  // Format 2: direct (x1,y1,x2,y2,score,class)
  if(A==6){
    for(int i=0;i<N;++i){
      const float* p = data + (size_t)i*A;
      float x1=p[0], y1=p[1], x2=p[2], y2=p[3], score=p[4]; int cls = (int)std::round(p[5]);
      if(score < thresh) continue;
      dets.push_back({score,x1,y1,x2,y2,cls,score});
    }
    std::sort(dets.begin(), dets.end(), [](auto&a, auto&b){return a.score>b.score;});
    if(dets.size()>50) dets.resize(50);
    return dets;
  }
  // Format 1: (cx,cy,w,h,obj,class_probs...)
  int num_cls = A-5;
  for(int i=0;i<N;++i){
    const float* p = data + (size_t)i*A;
    float cx=p[0], cy=p[1], w=p[2], h=p[3]; float obj=p[4]; if(obj<=0.f) continue;
    int best_c=-1; float best_v=0.f; for(int c=0;c<num_cls;++c){ float v=p[5+c]; if(v>best_v){best_v=v; best_c=c;} }
    float conf = obj * best_v; if(conf < thresh) continue;
    float x1 = cx - w/2.f; float y1 = cy - h/2.f; float x2 = cx + w/2.f; float y2 = cy + h/2.f;
    dets.push_back({conf,x1,y1,x2,y2,best_c,obj});
  }
  std::sort(dets.begin(), dets.end(), [](auto&a, auto&b){return a.score>b.score;});
  if(dets.size()>50) dets.resize(50);
  return dets;
}

// Compute IoU
static float IoU(const YoloDet& a, const YoloDet& b){
  float xx1 = std::max(a.x1,b.x1); float yy1 = std::max(a.y1,b.y1);
  float xx2 = std::min(a.x2,b.x2); float yy2 = std::min(a.y2,b.y2);
  float w = std::max(0.f, xx2-xx1); float h = std::max(0.f, yy2-yy1);
  float inter = w*h;
  float areaA = std::max(0.f,a.x2-a.x1)*std::max(0.f,a.y2-a.y1);
  float areaB = std::max(0.f,b.x2-b.x1)*std::max(0.f,b.y2-b.y1);
  float uni = areaA + areaB - inter + 1e-6f;
  return inter/uni;
}

static std::vector<YoloDet> NMS(const std::vector<YoloDet>& dets, float iou_thresh){
  std::vector<YoloDet> keep;
  for(const auto& d: dets){
    bool drop=false; for(const auto& k: keep){ if(k.cls==d.cls && IoU(k,d) > iou_thresh){ drop=true; break; } }
    if(!drop) keep.push_back(d);
  }
  return keep;
}

// Print first few floats (debug)
static void PrintPreview(const float* d, size_t n){ size_t k=std::min<size_t>(8,n); for(size_t i=0;i<k;++i){ if(i) std::cout<<' '; std::cout<<d[i]; } if(n>k) std::cout<<" ..."; std::cout<<"\n"; }

enum class ModelKind { ResNet, YOLO };
struct ModelSpec { std::string path; ModelKind kind; };

static void PrintLatencyStats(const char* tag, const std::vector<double>& ms){
  if(ms.empty()) return;
  double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
  double avg = sum / ms.size();
  auto [min_it, max_it] = std::minmax_element(ms.begin(), ms.end());
  std::cout << "[TIME] " << tag << ": min=" << *min_it << " ms, avg=" << avg << " ms, max=" << *max_it << " ms (n=" << ms.size() << ")\n";
}

static void RunResNet(Ort::Session& sess, Ort::MemoryInfo& mem, Ort::AllocatorWithDefaultOptions& alloc,
                      const std::string& image, const std::vector<std::string>& labels,
                      int warmup, int repeat){
  std::vector<int64_t> shape{1,3,224,224}; std::vector<float> pixels; bool ok=false; int ow=224, oh=224;
  if(!image.empty() && fs::exists(image)) ok = LoadImageToCHWFloat(image,224,224,pixels,true,&ow,&oh);
  if(!ok){ pixels.assign((size_t)3*224*224, 1.0f); std::cout << "[ResNet18] Warning: using padded fake image\n"; }
  auto input = Ort::Value::CreateTensor<float>(mem, pixels.data(), pixels.size(), shape.data(), shape.size());
  auto in_name = sess.GetInputNameAllocated(0, alloc); auto out_name = sess.GetOutputNameAllocated(0, alloc);
  const char* ins[] = { in_name.get() }; const char* outs[] = { out_name.get() };
  // Warmup
  for(int i=0;i<warmup;i++){ (void)sess.Run(Ort::RunOptions{}, ins, &input, 1, outs, 1); }
  // Timed runs (always on)
  if(repeat<=0) repeat=1;
  std::vector<double> times_ms; times_ms.reserve(repeat);
  Ort::Value last_out{nullptr};
  for(int i=0;i<repeat; ++i){
    auto t0 = std::chrono::steady_clock::now();
    auto out = sess.Run(Ort::RunOptions{}, ins, &input, 1, outs, 1);
    auto t1 = std::chrono::steady_clock::now();
    times_ms.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
    last_out = std::move(out.front());
  }
  PrintLatencyStats("ResNet18 Inference", times_ms);
  auto& t = last_out; float* logits = t.GetTensorMutableData<float>();
  size_t total=1; for(auto d: t.GetTensorTypeAndShapeInfo().GetShape()) if(d>0) total *= (size_t)d;
  std::vector<Ranked> top5; TopK(logits,total,5,top5);
  std::cout << "[ResNet18] Top-5:\n";
  for(auto& r: top5){ std::string label = (r.idx < (int)labels.size()? labels[r.idx] : std::string("<no-label>")); std::cout << "  #"<<r.idx<<"  prob="<<r.val<<"  "<<label<<"\n"; }
}

static void RunYOLO(Ort::Session& sess, Ort::MemoryInfo& mem, Ort::AllocatorWithDefaultOptions& alloc,
                    const std::string& image, float thresh, float nms_thresh,
                    const std::vector<std::string>& yolo_labels,
                    int warmup, int repeat){
  std::vector<int64_t> shape{1,3,640,640}; std::vector<float> pixels; bool ok=false; int orig_w=640, orig_h=640;
  if(!image.empty() && fs::exists(image)) ok = LoadImageToCHWFloat(image,640,640,pixels,false,&orig_w,&orig_h);
  if(!ok){ pixels.assign((size_t)3*640*640, 0.0f); std::cout << "[YOLO] Warning: using padded fake image\n"; orig_w=640; orig_h=640; }
  auto input = Ort::Value::CreateTensor<float>(mem, pixels.data(), pixels.size(), shape.data(), shape.size());
  auto in_name = sess.GetInputNameAllocated(0, alloc); auto out_name = sess.GetOutputNameAllocated(0, alloc);
  const char* ins[] = { in_name.get() }; const char* outs[] = { out_name.get() };
  // Warmup
  for(int i=0;i<warmup;i++){ (void)sess.Run(Ort::RunOptions{}, ins, &input, 1, outs, 1); }
  // Timed runs (always on)
  if(repeat<=0) repeat=1;
  std::vector<double> times_ms; times_ms.reserve(repeat);
  Ort::Value last_out{nullptr};
  for(int i=0;i<repeat; ++i){
    auto t0 = std::chrono::steady_clock::now();
    auto out = sess.Run(Ort::RunOptions{}, ins, &input, 1, outs, 1);
    auto t1 = std::chrono::steady_clock::now();
    times_ms.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
    last_out = std::move(out.front());
  }
  PrintLatencyStats("YOLO Inference", times_ms);

  auto& t = last_out; auto info = t.GetTensorTypeAndShapeInfo(); auto shape_out = info.GetShape();
  const float* data = t.GetTensorData<float>();
  if(shape_out.size()==3){
    int N = (int)shape_out[1]; int A = (int)shape_out[2];
    auto dets_raw = ParseYolo(data,N,A,thresh);
  // Rescale boxes back to original image coordinates and clamp
    float sx = static_cast<float>(orig_w)/640.f;
    float sy = static_cast<float>(orig_h)/640.f;
    for(auto& d: dets_raw){
      d.x1 = std::clamp(d.x1 * sx, 0.f, (float)(orig_w-1));
      d.x2 = std::clamp(d.x2 * sx, 0.f, (float)(orig_w-1));
      d.y1 = std::clamp(d.y1 * sy, 0.f, (float)(orig_h-1));
      d.y2 = std::clamp(d.y2 * sy, 0.f, (float)(orig_h-1));
    }
    auto dets = NMS(dets_raw, nms_thresh);
  std::cout << "[YOLO] raw_dets="<<dets_raw.size()<<", after_NMS="<<dets.size()<<" (conf>="<<thresh<<", nms="<<nms_thresh<<")\n";
    int show = std::min<size_t>(dets.size(), 10);
    for(int i=0;i<show;++i){
      auto& d=dets[i];
      std::string cname = (d.cls>=0 && d.cls < (int)yolo_labels.size()) ? yolo_labels[d.cls] : std::string("<cls="+std::to_string(d.cls)+">");
      std::cout << "  #"<<i
                << ": cls="<<d.cls
                << " ("<< cname << ")"
                << " conf="<<d.score
                << " box(xyxy)=["<<d.x1<<","<<d.y1<<","<<d.x2<<","<<d.y2<<"]\n";
    }
  if(dets.empty()) { std::cout << "  (no detection above threshold, preview raw:) "; PrintPreview(data, std::min(32, N*A)); }
  } else {
  std::cout << "[YOLO] Unrecognized output shape, preview first values: "; PrintPreview(data, 32);
  }
}

int main(int argc, char** argv){
  try {
  std::string select="all", image_path;
  int warmup=0, repeat=0;
  // Runtime default: labels/*.txt next to executable
  std::string exec_dir = GetExecutableDir();
  // Resource root: <prefix>/assets (executable in <prefix>/bin)
  std::string default_asset_root = (fs::path(exec_dir).parent_path()/"assets").string();
  // Allow environment override
  if(const char* envp = std::getenv("ORT_MODELS_ASSETS")){
    default_asset_root = envp;
  }
  std::string asset_root = default_asset_root; // Can be overridden by --asset-dir
  std::string labels_path = (fs::path(asset_root)/"labels"/"imagenet_classes.txt").string();
  std::string yolo_labels_path = (fs::path(asset_root)/"labels"/"coco_classes.txt").string();
  float yolo_thresh = 0.25f; float yolo_nms = 0.45f;
    for(int i=1;i<argc;i++){
      std::string a=argv[i];
      if(a=="--model" && i+1<argc){ select=argv[++i]; ToLower(select);} else
      if(a=="--image" && i+1<argc){ image_path=argv[++i]; } else
  if(a=="--labels" && i+1<argc){ labels_path=argv[++i]; } else
  if(a=="--yolo-labels" && i+1<argc){ yolo_labels_path=argv[++i]; } else
  if(a=="--yolo-thresh" && i+1<argc){ yolo_thresh = std::stof(argv[++i]); } else
  if(a=="--yolo-nms" && i+1<argc){ yolo_nms = std::stof(argv[++i]); } else
  if(a=="--asset-dir" && i+1<argc){ asset_root = argv[++i]; } else
  if(a=="--warmup" && i+1<argc){ warmup = std::stoi(argv[++i]); } else
  if(a=="--repeat" && i+1<argc){ repeat = std::stoi(argv[++i]); }
      else if(a=="-h"||a=="--help"){ Usage(); return 0; }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_models");
    Ort::SessionOptions opt; opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  const std::string model_dir = (fs::path(asset_root)/"models").string();
    std::vector<ModelSpec> models = {
      {model_dir+"/resnet-18.onnx", ModelKind::ResNet},
      {model_dir+"/yolo10s.onnx", ModelKind::YOLO}
    };
  std::vector<std::string> labels = fs::exists(labels_path)? ReadLines(labels_path) : std::vector<std::string>{};
  if(labels.empty()) std::cout << "[ResNet18] Warning: label file missing or empty: "<<labels_path<<"\n";
  std::vector<std::string> yolo_labels = fs::exists(yolo_labels_path)? ReadLines(yolo_labels_path) : std::vector<std::string>{};
  if(yolo_labels.empty()) std::cout << "[YOLO] Warning: COCO label file missing or empty: "<<yolo_labels_path<<" (will output class index only)\n";

    Ort::AllocatorWithDefaultOptions alloc; auto mem = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    for(auto& m: models){
      bool choose = (select=="all") ||
        (select=="resnet" && m.kind==ModelKind::ResNet) ||
        (select=="yolo" && m.kind==ModelKind::YOLO);
      if(!choose) continue;
  if(!fs::exists(m.path)){ std::cerr << "[SKIP] model missing: "<<m.path<<"\n"; continue; }
      std::cout << "\n[LOAD] "<<m.path<<"\n";
      Ort::Session sess(env, m.path.c_str(), opt);
      switch(m.kind){
        case ModelKind::ResNet: RunResNet(sess, mem, alloc, image_path, labels, warmup, repeat); break;
        case ModelKind::YOLO: RunYOLO(sess, mem, alloc, image_path, yolo_thresh, yolo_nms, yolo_labels, warmup, repeat); break;
      }
    }
  } catch(const Ort::Exception& e){ std::cerr << "ORT Exception: "<<e.what()<<"\n"; return 1; }
    catch(const std::exception& e){ std::cerr << "Exception: "<<e.what()<<"\n"; return 1; }
  return 0;
}
