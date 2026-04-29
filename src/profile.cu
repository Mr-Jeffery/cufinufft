#include <profile.h>

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define CUFINUFFT_HAVE_NVTX 1
#  elif __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>
#    define CUFINUFFT_HAVE_NVTX 1
#  else
#    define CUFINUFFT_HAVE_NVTX 0
#  endif
#else
#  define CUFINUFFT_HAVE_NVTX 0
#endif
#include <cstdio>


#if CUFINUFFT_HAVE_NVTX
const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 
	0x0000ffff, 0x00ff0000, 0x00ffffff }; 
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#endif

CudaTracer::CudaTracer(const char* name, int cid) 
{
#if CUFINUFFT_HAVE_NVTX
    PUSH_RANGE(name,cid);
#else
    (void)name;
    (void)cid;
#endif
}

CudaTracer::~CudaTracer() {
#if CUFINUFFT_HAVE_NVTX
    POP_RANGE;
#endif
}

