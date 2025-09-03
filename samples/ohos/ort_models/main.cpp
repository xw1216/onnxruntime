// ==============================================
// 多模型真实图片推理示例 (仅 ResNet18 + YOLO)
// - ResNet18: 读取图片, resize 到 224x224, 归一化 + ImageNet 均值方差, 输出 top-5 标签
// - YOLO: 读取图片, resize 到 640x640(简单拉伸), 解析第一输出张量, 打印若干检测框
// 依赖: third_party/stb_image.h, stb_image_resize2.h, imagenet_classes.txt
// 命令行:
//   --model <resnet|yolo|embed|all>
//   --image <image_path> (对 resnet / yolo 有效)
//   --labels <imagenet_label_file> (默认 ./third_party/imagenet_classes.txt)
//   --yolo-thresh <float> (默认 0.25)
// ==============================================

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <onnxruntime_cxx_api.h>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb_image_resize2.h"

namespace fs = std::filesystem;

static void Usage(){
  std::cout << "用法: ort_models --model <resnet|yolo|all> --image <path> [--labels file] [--yolo-thresh v] [--yolo-nms v]\n";
  std::cout << "说明: --yolo-thresh 置信度阈值(默认0.25)  --yolo-nms NMS IoU 阈值(默认0.45)\n";
}
static void ToLower(std::string& s){ for(auto& c:s) c=(char)std::tolower((unsigned char)c); }

// 读取全部行
static std::vector<std::string> ReadLines(const std::string& file){
  std::vector<std::string> lines; std::ifstream fin(file); std::string line; while(std::getline(fin,line)) if(!line.empty()) lines.push_back(line); return lines; }

// 简单 resize + 转 float (CHW)
static bool LoadImageToCHWFloat(const std::string& path, int target_w, int target_h, std::vector<float>& out, bool normalize, int* orig_w, int* orig_h){
  int w,h,c; unsigned char* data = stbi_load(path.c_str(), &w,&h,&c, 3); if(!data) return false; c=3; if(orig_w) *orig_w=w; if(orig_h) *orig_h=h;
  std::vector<unsigned char> resized((size_t)target_w*target_h*c);
  // 使用 stb resize, 输入 stride= w*c, 输出 stride= target_w*c
  if(w!=target_w || h!=target_h){
    stbir_resize_uint8_linear(data, w, h, w*c, resized.data(), target_w, target_h, target_w*c, STBIR_RGB);
  } else {
    std::memcpy(resized.data(), data, (size_t)w*h*c);
  }
  stbi_image_free(data);
  out.assign((size_t)c*target_w*target_h, 0.f);
  // 转 CHW + 0..1
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
  // 数值稳定: 找max
  float m = -1e30f; for(size_t i=0;i<n;++i) if(logits[i]>m) m=logits[i];
  std::vector<float> probs(n); float sum=0.f; for(size_t i=0;i<n;++i){ probs[i]=std::exp(logits[i]-m); sum += probs[i]; }
  for(size_t i=0;i<n;++i){ probs[i]/=sum; }
  for(size_t i=0;i<n;++i) out.push_back({(int)i, probs[i]});
  std::partial_sort(out.begin(), out.begin()+std::min<int>(k,out.size()), out.end(), [](const Ranked&a,const Ranked&b){return a.val>b.val;});
  if((int)out.size()>k) out.resize(k);
}

// YOLO 解析: 支持两种格式
// 1) [1,N,A], A>=6, 解释为 (cx,cy,w,h,obj,cls_probs...)
// 2) [1,N,6], 解释为 (x1,y1,x2,y2,score,class_id) —— 用户当前模型: shape=[1,300,6]
struct YoloDet { float score; float x1,y1,x2,y2; int cls; float obj; };
static std::vector<YoloDet> ParseYolo(const float* data, int N, int A, float thresh){
  std::vector<YoloDet> dets; if(A<6) return dets;
  // 格式2: 直接框 + score + class
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
  // 格式1: 转换中心点+宽高+obj+类别概率
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

// 计算 IoU
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

// 打印前几个 float (调试)
static void PrintPreview(const float* d, size_t n){ size_t k=std::min<size_t>(8,n); for(size_t i=0;i<k;++i){ if(i) std::cout<<' '; std::cout<<d[i]; } if(n>k) std::cout<<" ..."; std::cout<<"\n"; }

enum class ModelKind { ResNet, YOLO };
struct ModelSpec { std::string path; ModelKind kind; };

static void RunResNet(Ort::Session& sess, Ort::MemoryInfo& mem, Ort::AllocatorWithDefaultOptions& alloc, const std::string& image, const std::vector<std::string>& labels){
  std::vector<int64_t> shape{1,3,224,224}; std::vector<float> pixels; bool ok=false; int ow=224, oh=224;
  if(!image.empty() && fs::exists(image)) ok = LoadImageToCHWFloat(image,224,224,pixels,true,&ow,&oh);
  if(!ok){ pixels.assign((size_t)3*224*224, 1.0f); std::cout << "[ResNet18] 警告: 使用填充伪图像\n"; }
  auto input = Ort::Value::CreateTensor<float>(mem, pixels.data(), pixels.size(), shape.data(), shape.size());
  auto in_name = sess.GetInputNameAllocated(0, alloc); auto out_name = sess.GetOutputNameAllocated(0, alloc);
  const char* ins[] = { in_name.get() }; const char* outs[] = { out_name.get() };
  auto out = sess.Run(Ort::RunOptions{}, ins, &input, 1, outs, 1);
  auto& t = out.front(); float* logits = t.GetTensorMutableData<float>();
  size_t total=1; for(auto d: t.GetTensorTypeAndShapeInfo().GetShape()) if(d>0) total *= (size_t)d;
  std::vector<Ranked> top5; TopK(logits,total,5,top5);
  std::cout << "[ResNet18] Top-5:\n";
  for(auto& r: top5){ std::string label = (r.idx < (int)labels.size()? labels[r.idx] : std::string("<no-label>")); std::cout << "  #"<<r.idx<<"  prob="<<r.val<<"  "<<label<<"\n"; }
}

static void RunYOLO(Ort::Session& sess, Ort::MemoryInfo& mem, Ort::AllocatorWithDefaultOptions& alloc, const std::string& image, float thresh, float nms_thresh){
  std::vector<int64_t> shape{1,3,640,640}; std::vector<float> pixels; bool ok=false; int orig_w=640, orig_h=640;
  if(!image.empty() && fs::exists(image)) ok = LoadImageToCHWFloat(image,640,640,pixels,false,&orig_w,&orig_h);
  if(!ok){ pixels.assign((size_t)3*640*640, 0.0f); std::cout << "[YOLO] 警告: 使用填充伪图像\n"; orig_w=640; orig_h=640; }
  auto input = Ort::Value::CreateTensor<float>(mem, pixels.data(), pixels.size(), shape.data(), shape.size());
  auto in_name = sess.GetInputNameAllocated(0, alloc); auto out_name = sess.GetOutputNameAllocated(0, alloc);
  const char* ins[] = { in_name.get() }; const char* outs[] = { out_name.get() };
  auto results = sess.Run(Ort::RunOptions{}, ins, &input, 1, outs, 1);
  auto& t = results.front(); auto info = t.GetTensorTypeAndShapeInfo(); auto shape_out = info.GetShape();
  const float* data = t.GetTensorData<float>();
  if(shape_out.size()==3){
    int N = (int)shape_out[1]; int A = (int)shape_out[2];
    auto dets_raw = ParseYolo(data,N,A,thresh);
    // 反缩放回原图坐标 + 剪裁
    float sx = static_cast<float>(orig_w)/640.f;
    float sy = static_cast<float>(orig_h)/640.f;
    for(auto& d: dets_raw){
      d.x1 = std::clamp(d.x1 * sx, 0.f, (float)(orig_w-1));
      d.x2 = std::clamp(d.x2 * sx, 0.f, (float)(orig_w-1));
      d.y1 = std::clamp(d.y1 * sy, 0.f, (float)(orig_h-1));
      d.y2 = std::clamp(d.y2 * sy, 0.f, (float)(orig_h-1));
    }
    auto dets = NMS(dets_raw, nms_thresh);
    std::cout << "[YOLO] 原始检测="<<dets_raw.size()<<", NMS后="<<dets.size()<<" (conf>="<<thresh<<", nms="<<nms_thresh<<")\n";
    int show = std::min<size_t>(dets.size(), 10);
    for(int i=0;i<show;++i){ auto& d=dets[i];
      std::cout << "  #"<<i<<": cls="<<d.cls<<" conf="<<d.score
                <<" box(xyxy)= ["<<d.x1<<","<<d.y1<<","<<d.x2<<","<<d.y2<<"]\n"; }
    if(dets.empty()) { std::cout << "  (无满足阈值的检测, 原始输出预览:) "; PrintPreview(data, std::min(32, N*A)); }
  } else {
    std::cout << "[YOLO] 未识别的输出形状, 仅预览前几个值: "; PrintPreview(data, 32);
  }
}

int main(int argc, char** argv){
  try {
  std::string select="all", image_path, labels_path="./third_party/imagenet_classes.txt"; float yolo_thresh = 0.25f; float yolo_nms = 0.45f;
    for(int i=1;i<argc;i++){
      std::string a=argv[i];
      if(a=="--model" && i+1<argc){ select=argv[++i]; ToLower(select);} else
      if(a=="--image" && i+1<argc){ image_path=argv[++i]; } else
      if(a=="--labels" && i+1<argc){ labels_path=argv[++i]; } else
      if(a=="--yolo-thresh" && i+1<argc){ yolo_thresh = std::stof(argv[++i]); } else
      if(a=="--yolo-nms" && i+1<argc){ yolo_nms = std::stof(argv[++i]); }
      else if(a=="-h"||a=="--help"){ Usage(); return 0; }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_models");
    Ort::SessionOptions opt; opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const std::string model_dir = "./models"; // 与可执行同级
    std::vector<ModelSpec> models = {
      {model_dir+"/resnet-18.onnx", ModelKind::ResNet},
      {model_dir+"/yolo10s.onnx", ModelKind::YOLO}
    };
    std::vector<std::string> labels = fs::exists(labels_path)? ReadLines(labels_path) : std::vector<std::string>{};
    if(labels.empty()) std::cout << "[ResNet18] 警告: 标签文件缺失或为空: "<<labels_path<<"\n";

    Ort::AllocatorWithDefaultOptions alloc; auto mem = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    for(auto& m: models){
      bool choose = (select=="all") ||
        (select=="resnet" && m.kind==ModelKind::ResNet) ||
        (select=="yolo" && m.kind==ModelKind::YOLO);
      if(!choose) continue;
      if(!fs::exists(m.path)){ std::cerr << "[SKIP] 缺少模型: "<<m.path<<"\n"; continue; }
      std::cout << "\n[LOAD] "<<m.path<<"\n";
      Ort::Session sess(env, m.path.c_str(), opt);
      switch(m.kind){
        case ModelKind::ResNet: RunResNet(sess, mem, alloc, image_path, labels); break;
        case ModelKind::YOLO: RunYOLO(sess, mem, alloc, image_path, yolo_thresh, yolo_nms); break;
      }
    }
  } catch(const Ort::Exception& e){ std::cerr << "ORT Exception: "<<e.what()<<"\n"; return 1; }
    catch(const std::exception& e){ std::cerr << "Exception: "<<e.what()<<"\n"; return 1; }
  return 0;
}
