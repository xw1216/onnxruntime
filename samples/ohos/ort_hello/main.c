#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void CheckStatus(const OrtApi* api, OrtStatus* st) {
  if (st) {
    const char* msg = api->GetErrorMessage(st);
    fprintf(stderr, "ORT Error: %s\n", msg);
    api->ReleaseStatus(st);
    exit(1);
  }
}

int main() {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env = NULL;
  CheckStatus(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ohos-demo", &env));
  printf("Env created.\n");

  OrtSessionOptions* so = NULL;
  CheckStatus(api, api->CreateSessionOptions(&so));
  api->SetIntraOpNumThreads(so, 1); // ignore return (deprecated attr warning OK for demo)
  CheckStatus(api, api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_EXTENDED));
  printf("SessionOptions ready.\n");

  const char* model_path = "model.onnx"; // user should deploy a simple model next to binary
  OrtSession* session = NULL;
  OrtStatus* st = api->CreateSession(env, model_path, so, &session);
  if (st) {
    fprintf(stderr, "Load model '%s' failed (expected if model not present).\n", model_path);
    api->ReleaseStatus(st);
    printf("Demo finished without running inference.\n");
  } else {
    printf("Model loaded. (Inference code can be added here)\n");
    api->ReleaseSession(session);
  }

  api->ReleaseSessionOptions(so);
  api->ReleaseEnv(env);
  return 0;
}
