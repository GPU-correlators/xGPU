#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <nvml.h>
#include <pthread.h>
#include <signal.h>

#define NVML_CHECK(func)						\
  {									\
    nvmlReturn_t ret = func;						\
    if (ret != NVML_SUCCESS) {						\
      printf("Error %s returns %s\n", #func, nvmlErrorString(ret));	\
      exit(-1);								\
    }									\
  }

nvmlDevice_t GPUmon_device_id;

pthread_t GPUmon_thread;

FILE *GPUmon_out;

void* GPUmonitor(void *) {

  while (1) {
    // get power
    unsigned int power;
    NVML_CHECK(nvmlDeviceGetPowerUsage(GPUmon_device_id, &power));

    // get temperature
    unsigned int temp;
    NVML_CHECK(nvmlDeviceGetTemperature(GPUmon_device_id, NVML_TEMPERATURE_GPU, &temp));

    // get GPU clock
    unsigned int graphics_clock;
    NVML_CHECK(nvmlDeviceGetClockInfo(GPUmon_device_id, NVML_CLOCK_GRAPHICS, &graphics_clock));

    // get SM clock
    unsigned int sm_clock;
    NVML_CHECK(nvmlDeviceGetClockInfo(GPUmon_device_id, NVML_CLOCK_SM, &sm_clock));

    // get memory clock
    unsigned int memory_clock;
    NVML_CHECK(nvmlDeviceGetClockInfo(GPUmon_device_id, NVML_CLOCK_MEM, &memory_clock));

    // get time stamp
    clock_t time = clock();

    fprintf(GPUmon_out, "clock = %ld time = %e Power = %g watts, temperature = %u, clocks: graphics = %u sm = %u memory = %u\n",
	    time, (double)time/CLOCKS_PER_SEC, 1e-3*(float)power, temp, graphics_clock, sm_clock, memory_clock);

    usleep(10000); // sleep for 10 milliseconds
  }

  return NULL;
}

void GPUmonitorInit(int device_ordinal) {
  NVML_CHECK(nvmlInit());

  //unsigned int device_count;
  //NVML_CHECK(nvmlDeviceGetCount(&device_count));
  //printf("NVML: device count = %d\n", device_count);

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

  // spawn monitoring thread
  if (pthread_create(&GPUmon_thread, NULL, GPUmonitor, 0)) {
    printf("Failed to spawn thread in %s", __func__);
    exit(-1);
  }
}

void GPUmonitorFree() {

  // rejoin monitoring thread
  if (pthread_cancel(GPUmon_thread)) {
    printf("Failed to kill  thread in %s", __func__);
    exit(-1);
  }

  fclose(GPUmon_out);

  // shutdown NVML
  NVML_CHECK(nvmlShutdown());
}


