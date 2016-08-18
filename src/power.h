#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <nvml.h>
#include <pthread.h>
#include <signal.h>

// compute number of CUDA cores per SM
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x60, 64},  // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

#define NVML_CHECK(func)						\
  {									\
    nvmlReturn_t ret = func;						\
    if (ret == NVML_ERROR_NOT_SUPPORTED) {				\
      printf("%s not supported on this GPU\n", #func);			\
    } else if (ret != NVML_SUCCESS) {					\
      printf("Error %s returns %s\n", #func, nvmlErrorString(ret));	\
      exit(-1);								\
    }									\
  }

nvmlDevice_t GPUmon_device_id;

pthread_t GPUmon_thread;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int check = 0;

FILE *GPUmon_out;

unsigned int GPUmon_power;
unsigned int GPUmon_temp;
unsigned int GPUmon_graphics_clock;
unsigned int GPUmon_sm_clock;
unsigned int GPUmon_memory_clock;

void* GPUmonitor(void *) {

  int loop = check;
  while (loop == 1) {
    // get power
    NVML_CHECK(nvmlDeviceGetPowerUsage(GPUmon_device_id, &GPUmon_power));

    // get temperature
    NVML_CHECK(nvmlDeviceGetTemperature(GPUmon_device_id, NVML_TEMPERATURE_GPU, &GPUmon_temp));

    // get GPU clock
    NVML_CHECK(nvmlDeviceGetClockInfo(GPUmon_device_id, NVML_CLOCK_GRAPHICS, &GPUmon_graphics_clock));

    // get SM clock
    NVML_CHECK(nvmlDeviceGetClockInfo(GPUmon_device_id, NVML_CLOCK_SM, &GPUmon_sm_clock));

    // get memory clock
    NVML_CHECK(nvmlDeviceGetClockInfo(GPUmon_device_id, NVML_CLOCK_MEM, &GPUmon_memory_clock));

    // get time stamp
    clock_t time = clock();

    fprintf(GPUmon_out, "clock = %ld time = %e Power = %g watts, temperature = %u, clocks: graphics = %u sm = %u memory = %u\n",
	    time, (double)time/CLOCKS_PER_SEC, 1e-3*(float)GPUmon_power, GPUmon_temp, GPUmon_graphics_clock, GPUmon_sm_clock, GPUmon_memory_clock);

    usleep(10000); // sleep for 10 milliseconds

    pthread_mutex_lock(&mutex);
    loop = check;
    pthread_mutex_unlock(&mutex);
  }

  return NULL;
}

void GPUmonitorInit(int device_ordinal) {
  NVML_CHECK(nvmlInit());

  NVML_CHECK(nvmlDeviceGetHandleByIndex(device_ordinal, &GPUmon_device_id));
  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  NVML_CHECK(nvmlDeviceGetName(GPUmon_device_id, name, NVML_DEVICE_NAME_BUFFER_SIZE));
  printf("Initializing GPU monitoring on device %d: %s\n", device_ordinal, name);

  // get time stamp for filename
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long int time = 1000*tv.tv_sec + tv.tv_usec;

  char filename[1024];
  sprintf(filename, "gpu_monitor_%ld.out", time);

  GPUmon_out = fopen(filename, "w");
  fprintf(GPUmon_out, "%s\n", name);
  fprintf(GPUmon_out, "NSTATION = %d\n", NSTATION);
  fprintf(GPUmon_out, "NFREQUENCY = %d\n", NFREQUENCY);
  fprintf(GPUmon_out, "NTIME = %d\n", NTIME);
  fprintf(GPUmon_out, "NTIME_PIPE = %d\n", NTIME_PIPE);

  check = 1;

  // spawn monitoring thread
  if (pthread_create(&GPUmon_thread, NULL, GPUmonitor, 0)) {
    printf("Failed to spawn thread in %s", __func__);
    exit(-1);
  }
}

void GPUmonitorFree() {

  cudaDeviceSynchronize();

  sleep(1); // sleep for one second before ending

  // safely end the monitoring thread
  pthread_mutex_lock(&mutex);
  check = 0;
  pthread_mutex_unlock(&mutex);

  // rejoin monitoring thread
  //if (pthread_cancel(GPUmon_thread)) {
  //printf("Failed to kill  thread in %s", __func__);
  //exit(-1);
  //}

  fclose(GPUmon_out);

  // shutdown NVML
  NVML_CHECK(nvmlShutdown());
}


