/* ============================================================================
 * EVOX NEURO-FUZZY VISUALIZATION ENGINE - ULTIMATE WORKING VERSION
 * Author: David Sousa Oliver
 * Company: Dahua Technologies
 * File: evox/src/main.c
 * Version: 1.0.0
 *
 * Compilation:
 *   gcc -std=c90 -D_GNU_SOURCE -DCL_TARGET_OPENCL_VERSION=300 -pthread \
 *       -O3 -ffast-math -march=native -mtune=native -mavx2 -mfma \
 *       -malign-data=32 -lnuma -lOpenCL -lGL -lGLU -lopenal -lSDL2 -lm \
 *       -o evox_viz src/main.c
 *
 * ============================================================================ */

/* ----------------------------------------------------------------------------
 * System Headers
 * ---------------------------------------------------------------------------- */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <assert.h>

/* POSIX threading */
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/sysinfo.h>

/* NUMA support */
#ifdef __GNUC__
#define inline __inline__
#endif
#include <numa.h>
#undef inline

/* SIMD intrinsics */
#ifdef __AVX2__
#include <immintrin.h>
#define HAVE_AVX2 1
#else
#define HAVE_AVX2 0
#endif

/* OpenCL 3.0 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

/* OpenGL - include base headers */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

/* OpenAL */
#include <AL/al.h>
#include <AL/alc.h>

/* SDL2 */
#include <SDL2/SDL.h>

/* ----------------------------------------------------------------------------
 * Define all OpenGL types explicitly (since some may be missing)
 * ------------------------------------------------------------------------- */

#ifndef APIENTRY
#define APIENTRY
#endif

#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif

/* Basic OpenGL 1.1 types */
#ifndef PFNGLCLEARPROC
typedef void (APIENTRYP PFNGLCLEARPROC)(GLbitfield mask);
#endif

#ifndef PFNGLCLEARCOLORPROC
typedef void (APIENTRYP PFNGLCLEARCOLORPROC)(GLfloat red, GLfloat green,
		GLfloat blue, GLfloat alpha);
#endif

#ifndef PFNGLCLEARDEPTHPROC
typedef void (APIENTRYP PFNGLCLEARDEPTHPROC)(GLdouble depth);
#endif

#ifndef PFNGLCLEARSTENCILPROC
typedef void (APIENTRYP PFNGLCLEARSTENCILPROC)(GLint s);
#endif

#ifndef PFNGLCOLORMASKPROC
typedef void (APIENTRYP PFNGLCOLORMASKPROC)(GLboolean red, GLboolean green,
		GLboolean blue, GLboolean alpha);
#endif

#ifndef PFNGLDEPTHMASKPROC
typedef void (APIENTRYP PFNGLDEPTHMASKPROC)(GLboolean flag);
#endif

#ifndef PFNGLENABLEPROC
typedef void (APIENTRYP PFNGLENABLEPROC)(GLenum cap);
#endif

#ifndef PFNGLDISABLEPROC
typedef void (APIENTRYP PFNGLDISABLEPROC)(GLenum cap);
#endif

#ifndef PFNGLVIEWPORTPROC
typedef void (APIENTRYP PFNGLVIEWPORTPROC)(GLint x, GLint y, GLsizei width,
		GLsizei height);
#endif

#ifndef PFNGLMATRIXMODEPROC
typedef void (APIENTRYP PFNGLMATRIXMODEPROC)(GLenum mode);
#endif

#ifndef PFNGLLOADIDENTITYPROC
typedef void (APIENTRYP PFNGLLOADIDENTITYPROC)(void);
#endif

#ifndef PFNGLPUSHMATRIXPROC
typedef void (APIENTRYP PFNGLPUSHMATRIXPROC)(void);
#endif

#ifndef PFNGLPOPMATRIXPROC
typedef void (APIENTRYP PFNGLPOPMATRIXPROC)(void);
#endif

/* OpenGL 1.5 buffer object types */
#ifndef PFNGLGENBUFFERSPROC
typedef void (APIENTRYP PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
#endif

#ifndef PFNGLBINDBUFFERPROC
typedef void (APIENTRYP PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
#endif

#ifndef PFNGLBUFFERDATAPROC
typedef void (APIENTRYP PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size,
		const void *data, GLenum usage);
#endif

#ifndef PFNGLDELETEBUFFERSPROC
typedef void (APIENTRYP PFNGLDELETEBUFFERSPROC)(GLsizei n,
		const GLuint *buffers);
#endif

/* OpenGL 2.0 shader types */
#ifndef PFNGLCREATESHADERPROC
typedef GLuint (APIENTRYP PFNGLCREATESHADERPROC)(GLenum type);
#endif

#ifndef PFNGLSHADERSOURCEPROC
typedef void (APIENTRYP PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count,
		const GLchar *const*string, const GLint *length);
#endif

#ifndef PFNGLCOMPILESHADERPROC
typedef void (APIENTRYP PFNGLCOMPILESHADERPROC)(GLuint shader);
#endif

#ifndef PFNGLCREATEPROGRAMPROC
typedef GLuint (APIENTRYP PFNGLCREATEPROGRAMPROC)(void);
#endif

#ifndef PFNGLATTACHSHADERPROC
typedef void (APIENTRYP PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
#endif

#ifndef PFNGLLINKPROGRAMPROC
typedef void (APIENTRYP PFNGLLINKPROGRAMPROC)(GLuint program);
#endif

#ifndef PFNGLUSEPROGRAMPROC
typedef void (APIENTRYP PFNGLUSEPROGRAMPROC)(GLuint program);
#endif

#ifndef PFNGLDELETESHADERPROC
typedef void (APIENTRYP PFNGLDELETESHADERPROC)(GLuint shader);
#endif

#ifndef PFNGLDELETEPROGRAMPROC
typedef void (APIENTRYP PFNGLDELETEPROGRAMPROC)(GLuint program);
#endif

#ifndef PFNGLGETSHADERIVPROC
typedef void (APIENTRYP PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname,
		GLint *params);
#endif

#ifndef PFNGLGETSHADERINFOLOGPROC
typedef void (APIENTRYP PFNGLGETSHADERINFOLOGPROC)(GLuint shader,
		GLsizei bufSize, GLsizei *length, GLchar *infoLog);
#endif

#ifndef PFNGLGETPROGRAMIVPROC
typedef void (APIENTRYP PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname,
		GLint *params);
#endif

#ifndef PFNGLGETPROGRAMINFOLOGPROC
typedef void (APIENTRYP PFNGLGETPROGRAMINFOLOGPROC)(GLuint program,
		GLsizei bufSize, GLsizei *length, GLchar *infoLog);
#endif

#ifndef PFNGLGETUNIFORMLOCATIONPROC
typedef GLint (APIENTRYP PFNGLGETUNIFORMLOCATIONPROC)(GLuint program,
		const GLchar *name);
#endif

#ifndef PFNGLUNIFORMMATRIX4FVPROC
typedef void (APIENTRYP PFNGLUNIFORMMATRIX4FVPROC)(GLint location,
		GLsizei count, GLboolean transpose, const GLfloat *value);
#endif

#ifndef PFNGLUNIFORM1FPROC
typedef void (APIENTRYP PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
#endif

#ifndef PFNGLENABLEVERTEXATTRIBARRAYPROC
typedef void (APIENTRYP PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
#endif

#ifndef PFNGLVERTEXATTRIBPOINTERPROC
typedef void (APIENTRYP PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size,
		GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
#endif

/* OpenGL 3.0 vertex array object types */
#ifndef PFNGLGENVERTEXARRAYSPROC
typedef void (APIENTRYP PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
#endif

#ifndef PFNGLBINDVERTEXARRAYPROC
typedef void (APIENTRYP PFNGLBINDVERTEXARRAYPROC)(GLuint array);
#endif

#ifndef PFNGLDELETEVERTEXARRAYSPROC
typedef void (APIENTRYP PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n,
		const GLuint *arrays);
#endif

/* Texture types */
#ifndef PFNGLGENTEXTURESPROC
typedef void (APIENTRYP PFNGLGENTEXTURESPROC)(GLsizei n, GLuint *textures);
#endif

#ifndef PFNGLBINDTEXTUREPROC
typedef void (APIENTRYP PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
#endif

#ifndef PFNGLTEXIMAGE2DPROC
typedef void (APIENTRYP PFNGLTEXIMAGE2DPROC)(GLenum target, GLint level,
		GLint internalformat, GLsizei width, GLsizei height, GLint border,
		GLenum format, GLenum type, const void *pixels);
#endif

#ifndef PFNGLTEXPARAMETERIPROC
typedef void (APIENTRYP PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname,
		GLint param);
#endif

#ifndef PFNGLDELETETEXTURESPROC
typedef void (APIENTRYP PFNGLDELETETEXTURESPROC)(GLsizei n,
		const GLuint *textures);
#endif

/* Drawing types */
#ifndef PFNGLDRAWARRAYSPROC
typedef void (APIENTRYP PFNGLDRAWARRAYSPROC)(GLenum mode, GLint first,
		GLsizei count);
#endif

#ifndef PFNGLDRAWELEMENTSPROC
typedef void (APIENTRYP PFNGLDRAWELEMENTSPROC)(GLenum mode, GLsizei count,
		GLenum type, const void *indices);
#endif

/* ----------------------------------------------------------------------------
 * OpenGL Function Pointers
 * ------------------------------------------------------------------------- */

/* Basic OpenGL 1.1 functions */
static PFNGLCLEARPROC glClear_ptr;
static PFNGLCLEARCOLORPROC glClearColor_ptr;
static PFNGLVIEWPORTPROC glViewport_ptr;
static PFNGLDRAWARRAYSPROC glDrawArrays_ptr;
static PFNGLDRAWELEMENTSPROC glDrawElements_ptr;

/* Deletion functions */
static PFNGLDELETEBUFFERSPROC glDeleteBuffers_ptr;
static PFNGLDELETESHADERPROC glDeleteShader_ptr;
static PFNGLDELETEPROGRAMPROC glDeleteProgram_ptr;
static PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays_ptr;
static PFNGLDELETETEXTURESPROC glDeleteTextures_ptr;

/* Extension functions */
static PFNGLGENVERTEXARRAYSPROC glGenVertexArrays_ptr;
static PFNGLBINDVERTEXARRAYPROC glBindVertexArray_ptr;
static PFNGLGENBUFFERSPROC glGenBuffers_ptr;
static PFNGLBINDBUFFERPROC glBindBuffer_ptr;
static PFNGLBUFFERDATAPROC glBufferData_ptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray_ptr;
static PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer_ptr;
static PFNGLCREATESHADERPROC glCreateShader_ptr;
static PFNGLSHADERSOURCEPROC glShaderSource_ptr;
static PFNGLCOMPILESHADERPROC glCompileShader_ptr;
static PFNGLCREATEPROGRAMPROC glCreateProgram_ptr;
static PFNGLATTACHSHADERPROC glAttachShader_ptr;
static PFNGLLINKPROGRAMPROC glLinkProgram_ptr;
static PFNGLUSEPROGRAMPROC glUseProgram_ptr;
static PFNGLGETSHADERIVPROC glGetShaderiv_ptr;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog_ptr;
static PFNGLGETPROGRAMIVPROC glGetProgramiv_ptr;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog_ptr;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation_ptr;
static PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv_ptr;
static PFNGLUNIFORM1FPROC glUniform1f_ptr;
static PFNGLGENTEXTURESPROC glGenTextures_ptr;
static PFNGLBINDTEXTUREPROC glBindTexture_ptr;
static PFNGLTEXIMAGE2DPROC glTexImage2D_ptr;
static PFNGLTEXPARAMETERIPROC glTexParameteri_ptr;

/* ----------------------------------------------------------------------------
 * Compiler Attributes
 * ------------------------------------------------------------------------- */

#ifdef __GNUC__
#define ALIGNAS(x) __attribute__((aligned(x)))
#define PACKED      __attribute__((packed))
#define UNUSED      __attribute__((unused))
#define HOT         __attribute__((hot))
#define COLD        __attribute__((cold))
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ALIGNAS(x)
#define PACKED
#define UNUSED
#define HOT
#define COLD
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

/* ----------------------------------------------------------------------------
 * System Constants
 * ------------------------------------------------------------------------- */

#define PROJECT_NAME            "EVOX Neuro-Fuzzy Visualization Engine"
#define PROJECT_VERSION         "1.0.0"

/* Memory and alignment */
#define CACHE_LINE_SIZE         64
#define SIMD_ALIGNMENT          32
#define PAGE_SIZE               4096

/* Processing limits */
#define MAX_TOKENS              128
#define MAX_RULES               64
#define MAX_DIMENSIONS          4
#define MAX_THREADS             16
#define MAX_VERTICES            8192

/* Fuzzy system parameters */
#define FUZZY_SET_SIZE          256
#define ENTROPY_THRESHOLD       0.15
#define RULE_STRENGTH_MIN       0.001
#define MEMBERSHIP_EPSILON      1e-10

/* Rendering constants */
#define WINDOW_WIDTH            1280
#define WINDOW_HEIGHT           720
#define FPS_TARGET              60
#define FRAME_TIME_US           (1000000 / FPS_TARGET)

/* ----------------------------------------------------------------------------
 * Forward Declarations
 * ------------------------------------------------------------------------- */

typedef struct TokenFuzzyState TokenFuzzyState;
typedef struct MandaniRule MandaniRule;
typedef struct WeightVector WeightVector;
typedef struct NeuroFuzzySystem NeuroFuzzySystem;
typedef struct RenderVertex RenderVertex;
typedef struct NUMAThreadContext NUMAThreadContext;
typedef struct OpenCLWrapper OpenCLWrapper;
typedef struct OpenGLRenderer OpenGLRenderer;
typedef struct AudioEngine AudioEngine;
typedef struct WindowManager WindowManager;

/* ----------------------------------------------------------------------------
 * Core Data Structures (32-byte aligned for AVX)
 * ------------------------------------------------------------------------- */

struct ALIGNAS(32) TokenFuzzyState {
	double membership_values[MAX_DIMENSIONS];
	double entropy_weights[MAX_DIMENSIONS];
	unsigned char token_count;
	unsigned char dimension;
	unsigned short sequence_id;
	double confidence;
	double timestamp;
};

struct ALIGNAS(32) MandaniRule {
	double antecedents[MAX_DIMENSIONS];
	double consequents[MAX_DIMENSIONS];
	double rule_strength;
	double entropy_contribution;
	double support;
	double confidence;
	unsigned int antecedent_count;
	unsigned int consequent_count;
};

struct ALIGNAS(32) WeightVector {
	double axis_weights[MAX_DIMENSIONS];
	double dimension_factors[MAX_DIMENSIONS];
	double combined_vector[MAX_DIMENSIONS];
	double magnitude;
	double direction[MAX_DIMENSIONS];
	unsigned int flags;
};

struct NeuroFuzzySystem {
	TokenFuzzyState *token_states;
	MandaniRule *rule_base;
	WeightVector *current_weights;
	size_t token_count;
	size_t rule_count;
	double global_entropy;
	int processing_dimensions;
	double inference_time;
	unsigned long long frame_counter;
	struct timespec last_update;
};

struct ALIGNAS(32) RenderVertex {
	float position[3];
	float color[4];
	float normal[3];
	float tex_coord[2];
};

struct ALIGNAS(CACHE_LINE_SIZE) NUMAThreadContext {
	pthread_t thread_id;
	int numa_node;
	cpu_set_t cpu_affinity;
	NeuroFuzzySystem *local_system;
	TokenFuzzyState *local_tokens;
	size_t token_start;
	size_t token_end;
	WeightVector *results_aligned;
	pthread_barrier_t *barrier;
	double thread_entropy;
	unsigned long long operations;
	volatile int completed;
};

struct OpenCLWrapper {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_kernel fuzzy_kernel;
	cl_kernel entropy_kernel;
	cl_mem buffer_tokens;
	cl_mem buffer_rules;
	cl_mem buffer_weights;
	size_t work_group_size;
	int initialized;
};

struct OpenGLRenderer {
	GLuint vao;
	GLuint vbo;
	GLuint ebo;
	GLuint shader_program;
	GLuint texture_bgra;
	GLint uniform_projection;
	GLint uniform_view;
	GLint uniform_model;
	GLint uniform_time;
	float projection[16];
	float view[16];
	float model[16];
	int width;
	int height;
	float bg_color[4];
	float time;
};

struct AudioEngine {
	ALCdevice *device;
	ALCcontext *context;
	ALuint source;
	ALuint buffer;
	ALfloat listener_pos[3];
	ALfloat listener_vel[3];
	ALfloat listener_ori[6];
	double base_frequency;
	double frequency_range;
	int initialized;
};

struct WindowManager {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int width;
	int height;
	int keyboard_state[SDL_NUM_SCANCODES];
	int mouse_buttons;
	double mouse_x;
	double mouse_y;
	int quit_requested;
	char input_buffer[256];
	size_t input_length;
	float camera_distance;
	float camera_rotation;
};

/* ----------------------------------------------------------------------------
 * Function Prototypes
 * ------------------------------------------------------------------------- */

/* Initialization and cleanup */
int initialize_system(NeuroFuzzySystem **system);
void cleanup_system(NeuroFuzzySystem *system);
int initialize_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
		AudioEngine *audio, WindowManager *wm);
void cleanup_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
		AudioEngine *audio, WindowManager *wm);

/* NUMA thread functions */
void* numa_thread_worker(void *arg);
int create_numa_threads(NUMAThreadContext **contexts, int num_threads,
		NeuroFuzzySystem *system, pthread_barrier_t *barrier);
void join_numa_threads(NUMAThreadContext **contexts, int num_threads);
void bind_thread_to_numa_node(pthread_t thread, int numa_node);

/* Fuzzy inference */
double compute_membership(double input, double center, double width);
double evaluate_rule(const MandaniRule *rule, const double *inputs, int dim);
double compute_entropy(const double *weights, int count);
int optimize_weights(WeightVector *weights, double target_entropy);

/* OpenCL functions */
int init_opencl(OpenCLWrapper *ocl);
void cleanup_opencl(OpenCLWrapper *ocl);
int run_fuzzy_kernel(OpenCLWrapper *ocl, const TokenFuzzyState *tokens,
		size_t num_tokens, WeightVector *results);
int run_entropy_kernel(OpenCLWrapper *ocl, const double *weights, size_t count,
		double *entropy);

/* OpenGL functions */
int init_opengl(OpenGLRenderer *renderer, int width, int height);
void cleanup_opengl(OpenGLRenderer *renderer);
void render_cad_wireframe(OpenGLRenderer *renderer,
		const RenderVertex *vertices, size_t vertex_count,
		const unsigned int *indices, size_t index_count, float time);
void update_projection_matrix(float *matrix, float fov, float aspect,
		float near, float far);
void update_view_matrix(float *matrix, float distance, float rotation);
void generate_rich_visualization(const WeightVector *weights, size_t count,
		RenderVertex *vertices, unsigned int *indices, size_t *vertex_count,
		size_t *index_count, float time);

/* Audio functions */
int init_audio(AudioEngine *audio);
void cleanup_audio(AudioEngine *audio);
void play_spatial_audio(AudioEngine *audio, double frequency, double pan,
		double volume);

/* Window management */
int init_window(WindowManager *wm, int width, int height, const char *title);
void cleanup_window(WindowManager *wm);
int process_events(WindowManager *wm);
void handle_keyboard_input(WindowManager *wm, SDL_Keycode key);

/* Core pipeline functions */
int tokenize_text(const char *text, TokenFuzzyState *tokens, size_t max_tokens);
void fuzzy_inference(TokenFuzzyState *tokens, const MandaniRule *rules,
		size_t num_tokens, size_t num_rules);
void entropy_optimization(TokenFuzzyState *tokens, WeightVector *weights,
		double target_entropy);
void vector_compute(WeightVector *weights, const TokenFuzzyState *tokens,
		size_t count);

/* Utility functions */
double get_timestamp(void);
void sleep_us(long microseconds);
void log_error(const char *format, ...);
void log_info(const char *format, ...);
void* aligned_alloc_wrapper(size_t alignment, size_t size);
void aligned_free_wrapper(void *ptr);

/* ----------------------------------------------------------------------------
 * Global State
 * ------------------------------------------------------------------------- */

static ALIGNAS(CACHE_LINE_SIZE) struct {
		NeuroFuzzySystem *system;
		OpenCLWrapper ocl;
		OpenGLRenderer gl;
		AudioEngine audio;
		WindowManager wm;
		volatile int running;
		double global_start_time;
	} g_state = { 0 };

	/* ----------------------------------------------------------------------------
	 * Utility Functions
	 * ------------------------------------------------------------------------- */

	double get_timestamp(void) {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
	}

	void sleep_us(long microseconds) {
		struct timespec ts;
		ts.tv_sec = microseconds / 1000000;
		ts.tv_nsec = (microseconds % 1000000) * 1000;
		nanosleep(&ts, NULL);
	}

	void log_error(const char *format, ...) {
		va_list args;
		fprintf(stderr, "\033[31m[ERROR]\033[0m ");
		va_start(args, format);
		vfprintf(stderr, format, args);
		va_end(args);
		fprintf(stderr, "\n");
		fflush(stderr);
	}

	void log_info(const char *format, ...) {
		va_list args;
		printf("\033[32m[INFO]\033[0m ");
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		printf("\n");
		fflush(stdout);
	}

	void* aligned_alloc_wrapper(size_t alignment, size_t size) {
		void *ptr = NULL;
#ifdef _POSIX_VERSION
		if (posix_memalign(&ptr, alignment, size) == 0) {
			return ptr;
		}
#else
    ptr = malloc(size + alignment);
    if (ptr) {
        uintptr_t aligned = ((uintptr_t)ptr + alignment) & ~(alignment - 1);
        ((void**)aligned)[-1] = ptr;
        return (void*)aligned;
    }
#endif
		return NULL;
	}

	void aligned_free_wrapper(void *ptr) {
		if (!ptr)
			return;
#ifdef _POSIX_VERSION
		free(ptr);
#else
    void* original = ((void**)ptr)[-1];
    free(original);
#endif
	}

	/* ----------------------------------------------------------------------------
	 * Fuzzy Core Functions
	 * ------------------------------------------------------------------------- */

	double compute_membership(double input, double center, double width) {
		double diff = fabs(input - center);
		if (diff <= width) {
			return 1.0 - (diff / width);
		}
		return 0.0;
	}

	double evaluate_rule(const MandaniRule *rule, const double *inputs, int dim) {
		double activation = 1.0;
		int i;
		if (!rule || !inputs || dim <= 0)
			return 0.0;
		for (i = 0; i < dim; i++) {
			activation *= rule->antecedents[i] * inputs[i];
		}
		return activation;
	}

	double compute_entropy(const double *weights, int count) {
		double entropy = 0.0;
		int i;
		if (!weights || count <= 0)
			return 0.0;
		for (i = 0; i < count; i++) {
			double w = weights[i];
			if (w > MEMBERSHIP_EPSILON && w < 1.0 - MEMBERSHIP_EPSILON) {
				entropy -= w * log(w) + (1.0 - w) * log(1.0 - w);
			}
		}
		return entropy / log(2.0);
	}

	int optimize_weights(WeightVector *weights, double target_entropy) {
		double current_entropy = 0.0;
		double scale_factor;
		int i, j;
		if (!weights)
			return -1;

		for (i = 0; i < MAX_DIMENSIONS; i++) {
			double w = weights->combined_vector[i];
			if (w > MEMBERSHIP_EPSILON && w < 1.0 - MEMBERSHIP_EPSILON) {
				current_entropy -= w * log(w) + (1.0 - w) * log(1.0 - w);
			}
		}
		current_entropy /= log(2.0);

		if (fabs(current_entropy - target_entropy) > 0.01) {
			scale_factor = target_entropy
					/ (current_entropy + MEMBERSHIP_EPSILON);
			for (i = 0; i < MAX_DIMENSIONS; i++) {
				for (j = 0; j < MAX_DIMENSIONS; j++) {
					weights->axis_weights[j] *= scale_factor;
					weights->dimension_factors[j] *= scale_factor;
					weights->combined_vector[j] *= scale_factor;
				}
			}
		}
		return 0;
	}

	/* ----------------------------------------------------------------------------
	 * NUMA Thread Functions
	 * ------------------------------------------------------------------------- */

	void bind_thread_to_numa_node(pthread_t thread, int numa_node) {
		cpu_set_t cpuset;
		struct bitmask *node_cpus;
		int cpu;

		if (!numa_available() || numa_node < 0)
			return;

		node_cpus = numa_allocate_cpumask();
		if (numa_node_to_cpus(numa_node, node_cpus) < 0) {
			numa_free_cpumask(node_cpus);
			return;
		}

		CPU_ZERO(&cpuset);
		for (cpu = 0; cpu < CPU_SETSIZE; cpu++) {
			if (numa_bitmask_isbitset(node_cpus, cpu)) {
				CPU_SET(cpu, &cpuset);
			}
		}

		pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
		numa_set_preferred(numa_node);
		numa_free_cpumask(node_cpus);
	}

	void* numa_thread_worker(void *arg) {
		NUMAThreadContext *ctx = (NUMAThreadContext*) arg;
		size_t i, j;
		double local_entropy = 0.0;

		bind_thread_to_numa_node(pthread_self(), ctx->numa_node);

		ctx->thread_entropy = 0.0;
		ctx->operations = 0;
		ctx->completed = 0;

		if (ctx->barrier)
			pthread_barrier_wait(ctx->barrier);

		for (i = ctx->token_start; i < ctx->token_end; i++) {
			TokenFuzzyState *token = &ctx->local_tokens[i];
			WeightVector *result = &ctx->results_aligned[i];

			for (j = 0; j < MAX_DIMENSIONS; j++) {
				result->axis_weights[j] = token->membership_values[j];
				result->dimension_factors[j] = token->entropy_weights[j];
				result->combined_vector[j] = token->membership_values[j]
						* token->entropy_weights[j];
				local_entropy +=
						result->combined_vector[j]
								* (1.0 - result->combined_vector[j]
										+ MEMBERSHIP_EPSILON);
			}

			double mag_sq = 0.0;
			for (j = 0; j < MAX_DIMENSIONS; j++) {
				mag_sq += result->combined_vector[j]
						* result->combined_vector[j];
			}
			result->magnitude = sqrt(mag_sq);

			if (result->magnitude > MEMBERSHIP_EPSILON) {
				double inv_mag = 1.0 / result->magnitude;
				for (j = 0; j < MAX_DIMENSIONS; j++) {
					result->direction[j] = result->combined_vector[j] * inv_mag;
				}
			}

			result->flags = (token->token_count << 16) | token->dimension;
			ctx->operations++;
		}

		ctx->thread_entropy = local_entropy
				/ (ctx->token_end - ctx->token_start);
		if (ctx->barrier)
			pthread_barrier_wait(ctx->barrier);
		ctx->completed = 1;
		return NULL;
	}

	int create_numa_threads(NUMAThreadContext **contexts, int num_threads,
			NeuroFuzzySystem *system, pthread_barrier_t *barrier) {
		int i, j, ret;
		int numa_nodes = numa_num_configured_nodes();
		pthread_attr_t attr;

		if (!contexts || !system || num_threads <= 0)
			return -1;

		*contexts = (NUMAThreadContext*) aligned_alloc_wrapper(
		CACHE_LINE_SIZE, num_threads * sizeof(NUMAThreadContext));
		if (!*contexts) {
			log_error("Failed to allocate NUMA thread contexts");
			return -1;
		}

		memset(*contexts, 0, num_threads * sizeof(NUMAThreadContext));

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		size_t tokens_per_thread = system->token_count / num_threads;
		size_t remainder = system->token_count % num_threads;

		for (i = 0; i < num_threads; i++) {
			NUMAThreadContext *ctx = &(*contexts)[i];

			ctx->numa_node = i % numa_nodes;
			ctx->local_system = system;
			ctx->local_tokens = system->token_states;
			ctx->results_aligned = system->current_weights;
			ctx->barrier = barrier;

			ctx->token_start = i * tokens_per_thread;
			ctx->token_end = (i + 1) * tokens_per_thread;
			if (i == num_threads - 1)
				ctx->token_end += remainder;

			CPU_ZERO(&ctx->cpu_affinity);

			ret = pthread_create(&ctx->thread_id, &attr, numa_thread_worker,
					ctx);
			if (ret != 0) {
				log_error("Failed to create thread %d: %s", i, strerror(ret));
				for (j = 0; j < i; j++) {
					pthread_cancel((*contexts)[j].thread_id);
					pthread_join((*contexts)[j].thread_id, NULL);
				}
				aligned_free_wrapper(*contexts);
				*contexts = NULL;
				pthread_attr_destroy(&attr);
				return -1;
			}
		}

		pthread_attr_destroy(&attr);
		return 0;
	}

	void join_numa_threads(NUMAThreadContext **contexts, int num_threads) {
		int i;
		if (!contexts || !*contexts || num_threads <= 0)
			return;
		for (i = 0; i < num_threads; i++) {
			pthread_join((*contexts)[i].thread_id, NULL);
		}
		aligned_free_wrapper(*contexts);
		*contexts = NULL;
	}

	/* ----------------------------------------------------------------------------
	 * OpenCL Functions
	 * ------------------------------------------------------------------------- */

	int init_opencl(OpenCLWrapper *ocl) {
		cl_uint num_platforms;
		cl_uint num_devices;
		cl_int err;
		char device_name[256];
		char vendor_name[256];

		if (!ocl)
			return -1;
		memset(ocl, 0, sizeof(OpenCLWrapper));

		err = clGetPlatformIDs(1, &ocl->platform, &num_platforms);
		if (err != CL_SUCCESS || num_platforms == 0) {
			log_error("No OpenCL platforms found");
			return -1;
		}

		err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_GPU, 1, &ocl->device,
				&num_devices);
		if (err != CL_SUCCESS || num_devices == 0) {
			err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_CPU, 1,
					&ocl->device, &num_devices);
			if (err != CL_SUCCESS) {
				log_error("No OpenCL devices found");
				return -1;
			}
		}

		clGetDeviceInfo(ocl->device, CL_DEVICE_NAME, sizeof(device_name),
				device_name, NULL);
		clGetDeviceInfo(ocl->device, CL_DEVICE_VENDOR, sizeof(vendor_name),
				vendor_name, NULL);
		log_info("OpenCL device: %s - %s", vendor_name, device_name);

		clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
				sizeof(ocl->work_group_size), &ocl->work_group_size, NULL);

		ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
		if (err != CL_SUCCESS) {
			log_error("Failed to create OpenCL context");
			return -1;
		}

		ocl->queue = clCreateCommandQueueWithProperties(ocl->context,
				ocl->device,
				NULL, &err);
		if (err != CL_SUCCESS) {
			log_error("Failed to create command queue");
			clReleaseContext(ocl->context);
			return -1;
		}

		const char *kernel_source = "__kernel void fuzzy_inference(\n"
				"    __global const double* tokens,\n"
				"    __global double* results,\n"
				"    const unsigned int num_tokens\n"
				") {\n"
				"    unsigned int token_idx = get_global_id(0);\n"
				"    if (token_idx >= num_tokens) return;\n"
				"    \n"
				"    double membership[4];\n"
				"    double entropy[4];\n"
				"    \n"
				"    for (int i = 0; i < 4; i++) {\n"
				"        membership[i] = tokens[token_idx * 8 + i];\n"
				"        entropy[i] = tokens[token_idx * 8 + i + 4];\n"
				"    }\n"
				"    \n"
				"    double combined[4];\n"
				"    double magnitude = 0.0;\n"
				"    for (int i = 0; i < 4; i++) {\n"
				"        combined[i] = membership[i] * entropy[i];\n"
				"        magnitude += combined[i] * combined[i];\n"
				"    }\n"
				"    magnitude = sqrt(magnitude);\n"
				"    \n"
				"    for (int i = 0; i < 4; i++) {\n"
				"        results[token_idx * 8 + i] = membership[i];\n"
				"        results[token_idx * 8 + i + 4] = combined[i];\n"
				"    }\n"
				"}\n";

		cl_program program = clCreateProgramWithSource(ocl->context, 1,
				&kernel_source, NULL, &err);
		if (err != CL_SUCCESS) {
			log_error("Failed to create OpenCL program");
			clReleaseCommandQueue(ocl->queue);
			clReleaseContext(ocl->context);
			return -1;
		}

		err = clBuildProgram(program, 1, &ocl->device, "-cl-fast-relaxed-math",
		NULL, NULL);
		if (err != CL_SUCCESS) {
			char build_log[16384];
			clGetProgramBuildInfo(program, ocl->device, CL_PROGRAM_BUILD_LOG,
					sizeof(build_log), build_log, NULL);
			log_error("OpenCL build failed:\n%s", build_log);
			clReleaseProgram(program);
			clReleaseCommandQueue(ocl->queue);
			clReleaseContext(ocl->context);
			return -1;
		}

		ocl->fuzzy_kernel = clCreateKernel(program, "fuzzy_inference", &err);
		if (err != CL_SUCCESS) {
			log_error("Failed to create fuzzy kernel");
			clReleaseProgram(program);
			clReleaseCommandQueue(ocl->queue);
			clReleaseContext(ocl->context);
			return -1;
		}

		ocl->buffer_tokens = NULL;
		ocl->buffer_rules = NULL;
		ocl->buffer_weights = NULL;

		clReleaseProgram(program);
		ocl->initialized = 1;
		return 0;
	}

	void cleanup_opencl(OpenCLWrapper *ocl) {
		if (!ocl || !ocl->initialized)
			return;

		if (ocl->fuzzy_kernel)
			clReleaseKernel(ocl->fuzzy_kernel);
		if (ocl->entropy_kernel)
			clReleaseKernel(ocl->entropy_kernel);
		if (ocl->buffer_tokens)
			clReleaseMemObject(ocl->buffer_tokens);
		if (ocl->buffer_rules)
			clReleaseMemObject(ocl->buffer_rules);
		if (ocl->buffer_weights)
			clReleaseMemObject(ocl->buffer_weights);
		if (ocl->queue)
			clReleaseCommandQueue(ocl->queue);
		if (ocl->context)
			clReleaseContext(ocl->context);

		memset(ocl, 0, sizeof(OpenCLWrapper));
	}

	int run_fuzzy_kernel(UNUSED OpenCLWrapper *ocl,
	UNUSED const TokenFuzzyState *tokens,
	UNUSED size_t num_tokens,
	UNUSED WeightVector *results) {
		return 0;
	}

	int run_entropy_kernel(UNUSED OpenCLWrapper *ocl,
	UNUSED const double *weights,
	UNUSED size_t count,
	UNUSED double *entropy) {
		return 0;
	}

	/* ----------------------------------------------------------------------------
	 * OpenGL Functions
	 * ------------------------------------------------------------------------- */

	/* Helper function to load OpenGL functions */
	static void* get_gl_function(const char *name) {
		return glXGetProcAddress((const GLubyte*) name);
	}

	void update_projection_matrix(float *matrix, float fov, float aspect,
			float near, float far) {
		float tan_half_fov = tanf(fov * 0.5f * 3.14159f / 180.0f);
		memset(matrix, 0, 16 * sizeof(float));
		matrix[0] = 1.0f / (aspect * tan_half_fov);
		matrix[5] = 1.0f / tan_half_fov;
		matrix[10] = -(far + near) / (far - near);
		matrix[11] = -1.0f;
		matrix[14] = -(2.0f * far * near) / (far - near);
	}

	void update_view_matrix(float *matrix, float distance, float rotation) {
		memset(matrix, 0, 16 * sizeof(float));
		matrix[0] = cosf(rotation);
		matrix[2] = sinf(rotation);
		matrix[5] = 1.0f;
		matrix[8] = -sinf(rotation);
		matrix[10] = cosf(rotation);
		matrix[14] = -distance;
		matrix[15] = 1.0f;
	}

	int init_opengl(OpenGLRenderer *renderer, int width, int height) {
		if (!renderer)
			return -1;

		memset(renderer, 0, sizeof(OpenGLRenderer));
		renderer->width = width;
		renderer->height = height;
		renderer->time = 0.0f;

		/* Load basic OpenGL 1.1 functions */
		glClear_ptr = (PFNGLCLEARPROC) get_gl_function("glClear");
		glClearColor_ptr = (PFNGLCLEARCOLORPROC) get_gl_function(
				"glClearColor");
		glViewport_ptr = (PFNGLVIEWPORTPROC) get_gl_function("glViewport");
		glDrawArrays_ptr = (PFNGLDRAWARRAYSPROC) get_gl_function(
				"glDrawArrays");
		glDrawElements_ptr = (PFNGLDRAWELEMENTSPROC) get_gl_function(
				"glDrawElements");

		/* Load deletion functions */
		glDeleteBuffers_ptr = (PFNGLDELETEBUFFERSPROC) get_gl_function(
				"glDeleteBuffers");
		glDeleteShader_ptr = (PFNGLDELETESHADERPROC) get_gl_function(
				"glDeleteShader");
		glDeleteProgram_ptr = (PFNGLDELETEPROGRAMPROC) get_gl_function(
				"glDeleteProgram");
		glDeleteVertexArrays_ptr =
				(PFNGLDELETEVERTEXARRAYSPROC) get_gl_function(
						"glDeleteVertexArrays");
		glDeleteTextures_ptr = (PFNGLDELETETEXTURESPROC) get_gl_function(
				"glDeleteTextures");

		/* Load extension functions */
		glGenVertexArrays_ptr = (PFNGLGENVERTEXARRAYSPROC) get_gl_function(
				"glGenVertexArrays");
		glBindVertexArray_ptr = (PFNGLBINDVERTEXARRAYPROC) get_gl_function(
				"glBindVertexArray");
		glGenBuffers_ptr = (PFNGLGENBUFFERSPROC) get_gl_function(
				"glGenBuffers");
		glBindBuffer_ptr = (PFNGLBINDBUFFERPROC) get_gl_function(
				"glBindBuffer");
		glBufferData_ptr = (PFNGLBUFFERDATAPROC) get_gl_function(
				"glBufferData");
		glEnableVertexAttribArray_ptr =
				(PFNGLENABLEVERTEXATTRIBARRAYPROC) get_gl_function(
						"glEnableVertexAttribArray");
		glVertexAttribPointer_ptr =
				(PFNGLVERTEXATTRIBPOINTERPROC) get_gl_function(
						"glVertexAttribPointer");
		glCreateShader_ptr = (PFNGLCREATESHADERPROC) get_gl_function(
				"glCreateShader");
		glShaderSource_ptr = (PFNGLSHADERSOURCEPROC) get_gl_function(
				"glShaderSource");
		glCompileShader_ptr = (PFNGLCOMPILESHADERPROC) get_gl_function(
				"glCompileShader");
		glCreateProgram_ptr = (PFNGLCREATEPROGRAMPROC) get_gl_function(
				"glCreateProgram");
		glAttachShader_ptr = (PFNGLATTACHSHADERPROC) get_gl_function(
				"glAttachShader");
		glLinkProgram_ptr = (PFNGLLINKPROGRAMPROC) get_gl_function(
				"glLinkProgram");
		glUseProgram_ptr = (PFNGLUSEPROGRAMPROC) get_gl_function(
				"glUseProgram");
		glGetShaderiv_ptr = (PFNGLGETSHADERIVPROC) get_gl_function(
				"glGetShaderiv");
		glGetShaderInfoLog_ptr = (PFNGLGETSHADERINFOLOGPROC) get_gl_function(
				"glGetShaderInfoLog");
		glGetProgramiv_ptr = (PFNGLGETPROGRAMIVPROC) get_gl_function(
				"glGetProgramiv");
		glGetProgramInfoLog_ptr = (PFNGLGETPROGRAMINFOLOGPROC) get_gl_function(
				"glGetProgramInfoLog");
		glGetUniformLocation_ptr =
				(PFNGLGETUNIFORMLOCATIONPROC) get_gl_function(
						"glGetUniformLocation");
		glUniformMatrix4fv_ptr = (PFNGLUNIFORMMATRIX4FVPROC) get_gl_function(
				"glUniformMatrix4fv");
		glUniform1f_ptr = (PFNGLUNIFORM1FPROC) get_gl_function("glUniform1f");
		glGenTextures_ptr = (PFNGLGENTEXTURESPROC) get_gl_function(
				"glGenTextures");
		glBindTexture_ptr = (PFNGLBINDTEXTUREPROC) get_gl_function(
				"glBindTexture");
		glTexImage2D_ptr = (PFNGLTEXIMAGE2DPROC) get_gl_function(
				"glTexImage2D");
		glTexParameteri_ptr = (PFNGLTEXPARAMETERIPROC) get_gl_function(
				"glTexParameteri");

		if (!glGenVertexArrays_ptr || !glBindVertexArray_ptr
				|| !glGenBuffers_ptr) {
			log_error("Failed to load OpenGL function pointers");
			return -1;
		}

		/* Set background color to dark space blue */
		renderer->bg_color[0] = 0.05f;
		renderer->bg_color[1] = 0.05f;
		renderer->bg_color[2] = 0.1f;
		renderer->bg_color[3] = 1.0f;

		/* Generate buffers */
		glGenVertexArrays_ptr(1, &renderer->vao);
		glGenBuffers_ptr(1, &renderer->vbo);
		glGenBuffers_ptr(1, &renderer->ebo);

		/* Create shader program */
		const char *vertex_shader_src =
				"#version 330 core\n"
						"layout(location = 0) in vec3 position;\n"
						"layout(location = 1) in vec4 color;\n"
						"layout(location = 2) in vec3 normal;\n"
						"layout(location = 3) in vec2 texCoord;\n"
						"out vec4 fragColor;\n"
						"out vec3 fragNormal;\n"
						"out vec2 fragTexCoord;\n"
						"uniform mat4 projection;\n"
						"uniform mat4 view;\n"
						"uniform mat4 model;\n"
						"uniform float time;\n"
						"void main() {\n"
						"    vec3 pos = position;\n"
						"    pos.x += sin(time * 2.0 + position.y) * 0.1;\n"
						"    pos.y += cos(time * 1.5 + position.z) * 0.1;\n"
						"    pos.z += sin(time * 1.8 + position.x) * 0.1;\n"
						"    gl_Position = projection * view * model * vec4(pos, 1.0);\n"
						"    fragColor = color;\n"
						"    fragNormal = normal;\n"
						"    fragTexCoord = texCoord;\n"
						"}\n";

		const char *fragment_shader_src =
				"#version 330 core\n"
						"in vec4 fragColor;\n"
						"in vec3 fragNormal;\n"
						"in vec2 fragTexCoord;\n"
						"out vec4 outputColor;\n"
						"uniform float time;\n"
						"void main() {\n"
						"    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));\n"
						"    float diff = max(dot(fragNormal, lightDir), 0.2);\n"
						"    vec3 color = fragColor.rgb * diff;\n"
						"    color += vec3(sin(time + fragTexCoord.x * 10.0) * 0.1,\n"
						"                  cos(time + fragTexCoord.y * 10.0) * 0.1,\n"
						"                  sin(time * 0.5 + fragTexCoord.x * 5.0) * 0.1);\n"
						"    outputColor = vec4(color, fragColor.a);\n"
						"}\n";

		GLuint vertex_shader = glCreateShader_ptr(GL_VERTEX_SHADER);
		glShaderSource_ptr(vertex_shader, 1, &vertex_shader_src, NULL);
		glCompileShader_ptr(vertex_shader);

		GLuint fragment_shader = glCreateShader_ptr(GL_FRAGMENT_SHADER);
		glShaderSource_ptr(fragment_shader, 1, &fragment_shader_src, NULL);
		glCompileShader_ptr(fragment_shader);

		GLint success;
		glGetShaderiv_ptr(vertex_shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			char info_log[512];
			glGetShaderInfoLog_ptr(vertex_shader, sizeof(info_log), NULL,
					info_log);
			log_error("Vertex shader compilation failed: %s", info_log);
			return -1;
		}

		glGetShaderiv_ptr(fragment_shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			char info_log[512];
			glGetShaderInfoLog_ptr(fragment_shader, sizeof(info_log), NULL,
					info_log);
			log_error("Fragment shader compilation failed: %s", info_log);
			return -1;
		}

		renderer->shader_program = glCreateProgram_ptr();
		glAttachShader_ptr(renderer->shader_program, vertex_shader);
		glAttachShader_ptr(renderer->shader_program, fragment_shader);
		glLinkProgram_ptr(renderer->shader_program);

		glGetProgramiv_ptr(renderer->shader_program, GL_LINK_STATUS, &success);
		if (!success) {
			char info_log[512];
			glGetProgramInfoLog_ptr(renderer->shader_program, sizeof(info_log),
					NULL, info_log);
			log_error("Shader program linking failed: %s", info_log);
			return -1;
		}

		/* Clean up shaders */
		if (glDeleteShader_ptr) {
			glDeleteShader_ptr(vertex_shader);
			glDeleteShader_ptr(fragment_shader);
		}

		renderer->uniform_projection = glGetUniformLocation_ptr(
				renderer->shader_program, "projection");
		renderer->uniform_view = glGetUniformLocation_ptr(
				renderer->shader_program, "view");
		renderer->uniform_model = glGetUniformLocation_ptr(
				renderer->shader_program, "model");
		renderer->uniform_time = glGetUniformLocation_ptr(
				renderer->shader_program, "time");

		/* Create BGRA texture */
		if (glGenTextures_ptr && glBindTexture_ptr && glTexParameteri_ptr) {
			glGenTextures_ptr(1, &renderer->texture_bgra);
			glBindTexture_ptr(GL_TEXTURE_2D, renderer->texture_bgra);
			glTexParameteri_ptr(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
					GL_LINEAR);
			glTexParameteri_ptr(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
					GL_LINEAR);
		} else {
			renderer->texture_bgra = 0;
		}

		update_projection_matrix(renderer->projection, 45.0f,
				(float) width / (float) height, 0.1f, 100.0f);
		update_view_matrix(renderer->view, 8.0f, 0.0f);

		memset(renderer->model, 0, sizeof(renderer->model));
		renderer->model[0] = 1.0f;
		renderer->model[5] = 1.0f;
		renderer->model[10] = 1.0f;
		renderer->model[15] = 1.0f;

		return 0;
	}

	void cleanup_opengl(OpenGLRenderer *renderer) {
		if (!renderer)
			return;

		if (renderer->vbo && glDeleteBuffers_ptr)
			glDeleteBuffers_ptr(1, &renderer->vbo);
		if (renderer->ebo && glDeleteBuffers_ptr)
			glDeleteBuffers_ptr(1, &renderer->ebo);
		if (renderer->vao && glDeleteVertexArrays_ptr)
			glDeleteVertexArrays_ptr(1, &renderer->vao);
		if (renderer->texture_bgra && glDeleteTextures_ptr)
			glDeleteTextures_ptr(1, &renderer->texture_bgra);
		if (renderer->shader_program && glDeleteProgram_ptr)
			glDeleteProgram_ptr(renderer->shader_program);

		memset(renderer, 0, sizeof(OpenGLRenderer));
	}

	void render_cad_wireframe(OpenGLRenderer *renderer,
			const RenderVertex *vertices, size_t vertex_count,
			const unsigned int *indices, size_t index_count, float time) {
		if (!renderer || !vertices || vertex_count == 0)
			return;

		glBindVertexArray_ptr(renderer->vao);

		glBindBuffer_ptr(GL_ARRAY_BUFFER, renderer->vbo);
		glBufferData_ptr(GL_ARRAY_BUFFER, vertex_count * sizeof(RenderVertex),
				vertices, GL_DYNAMIC_DRAW);

		/* Position attribute */
		glEnableVertexAttribArray_ptr(0);
		glVertexAttribPointer_ptr(0, 3, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, position));

		/* Color attribute */
		glEnableVertexAttribArray_ptr(1);
		glVertexAttribPointer_ptr(1, 4, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, color));

		/* Normal attribute */
		glEnableVertexAttribArray_ptr(2);
		glVertexAttribPointer_ptr(2, 3, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, normal));

		/* TexCoord attribute */
		glEnableVertexAttribArray_ptr(3);
		glVertexAttribPointer_ptr(3, 2, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, tex_coord));

		if (indices && index_count > 0) {
			glBindBuffer_ptr(GL_ELEMENT_ARRAY_BUFFER, renderer->ebo);
			glBufferData_ptr(GL_ELEMENT_ARRAY_BUFFER,
					index_count * sizeof(unsigned int), indices,
					GL_DYNAMIC_DRAW);
		}

		glUseProgram_ptr(renderer->shader_program);
		glUniformMatrix4fv_ptr(renderer->uniform_projection, 1, GL_FALSE,
				renderer->projection);
		glUniformMatrix4fv_ptr(renderer->uniform_view, 1, GL_FALSE,
				renderer->view);
		glUniformMatrix4fv_ptr(renderer->uniform_model, 1, GL_FALSE,
				renderer->model);
		if (renderer->uniform_time >= 0)
			glUniform1f_ptr(renderer->uniform_time, time);

		if (glClearColor_ptr) {
			glClearColor_ptr(renderer->bg_color[0], renderer->bg_color[1],
					renderer->bg_color[2], renderer->bg_color[3]);
		}
		if (glClear_ptr) {
			glClear_ptr(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		if (indices && index_count > 0) {
			if (glDrawElements_ptr) {
				glDrawElements_ptr(GL_TRIANGLES, (GLsizei) index_count,
						GL_UNSIGNED_INT, NULL);
			}
		} else {
			if (glDrawArrays_ptr) {
				glDrawArrays_ptr(GL_TRIANGLES, 0, (GLsizei) vertex_count);
			}
		}
	}

	void generate_rich_visualization(const WeightVector *weights, size_t count,
			RenderVertex *vertices, unsigned int *indices, size_t *vertex_count,
			size_t *index_count, float time) {
		size_t i, j, k;
		size_t v_idx = 0;
		size_t i_idx = 0;

		if (!weights || !vertices || !indices || !vertex_count || !index_count)
			return;

		for (i = 0; i < count && v_idx < MAX_VERTICES - 36; i++) {
			if (weights[i].magnitude < MEMBERSHIP_EPSILON)
				continue;

			float size = (float) weights[i].magnitude * 2.0f;
			float hue = (float) ((weights[i].flags >> 16) & 0xFF) / 255.0f;
			float r = sinf(hue * 6.283f) * 0.5f + 0.5f;
			float g = sinf((hue + 0.33f) * 6.283f) * 0.5f + 0.5f;
			float b = sinf((hue + 0.67f) * 6.283f) * 0.5f + 0.5f;

			/* Position based on direction vector */
			float x = (float) weights[i].direction[0] * 3.0f;
			float y = (float) weights[i].direction[1] * 3.0f;
			float z = (float) weights[i].direction[2] * 3.0f;

			/* Create an icosahedron-inspired shape */
			float vertices_data[12][3] = { { x - size, y - size, z - size }, { x
					+ size, y - size, z - size },
					{ x + size, y + size, z - size }, { x - size, y + size, z
							- size }, { x - size, y - size, z + size }, { x
							+ size, y - size, z + size }, { x + size, y + size,
							z + size }, { x - size, y + size, z + size },

					/* Additional vertices for more complex geometry */
					{ x, y - size * 1.5f, z }, { x, y + size * 1.5f, z }, { x
							- size * 1.5f, y, z }, { x + size * 1.5f, y, z } };

			/* Create triangular faces */
			int faces[20][3] = { { 0, 1, 2 }, { 0, 2, 3 }, { 4, 6, 5 }, { 4, 7,
					6 }, { 0, 4, 5 }, { 0, 5, 1 }, { 1, 5, 6 }, { 1, 6, 2 }, {
					2, 6, 7 }, { 2, 7, 3 }, { 3, 7, 4 }, { 3, 4, 0 },
					{ 8, 0, 1 }, { 8, 1, 5 }, { 8, 5, 4 }, { 8, 4, 0 }, { 9, 2,
							3 }, { 9, 3, 7 }, { 9, 7, 6 }, { 9, 6, 2 } };

			for (j = 0; j < 20; j++) {
				for (k = 0; k < 3; k++) {
					int v = faces[j][k];
					vertices[v_idx + k].position[0] = vertices_data[v][0];
					vertices[v_idx + k].position[1] = vertices_data[v][1];
					vertices[v_idx + k].position[2] = vertices_data[v][2];

					vertices[v_idx + k].color[0] = r
							* (0.7f + 0.3f * sinf(time + (float) j));
					vertices[v_idx + k].color[1] = g
							* (0.7f + 0.3f * cosf(time + (float) j));
					vertices[v_idx + k].color[2] = b
							* (0.7f + 0.3f * sinf(time * 0.5f + (float) j));
					vertices[v_idx + k].color[3] = 0.8f;

					/* Compute normal */
					vertices[v_idx + k].normal[0] = vertices_data[v][0] - x;
					vertices[v_idx + k].normal[1] = vertices_data[v][1] - y;
					vertices[v_idx + k].normal[2] = vertices_data[v][2] - z;
					float len = sqrtf(
							vertices[v_idx + k].normal[0]
									* vertices[v_idx + k].normal[0]
									+ vertices[v_idx + k].normal[1]
											* vertices[v_idx + k].normal[1]
									+ vertices[v_idx + k].normal[2]
											* vertices[v_idx + k].normal[2]);
					if (len > 0) {
						vertices[v_idx + k].normal[0] /= len;
						vertices[v_idx + k].normal[1] /= len;
						vertices[v_idx + k].normal[2] /= len;
					}

					vertices[v_idx + k].tex_coord[0] = (float) k / 3.0f;
					vertices[v_idx + k].tex_coord[1] = (float) j / 20.0f;
				}

				indices[i_idx++] = (unsigned int) v_idx;
				indices[i_idx++] = (unsigned int) v_idx + 1;
				indices[i_idx++] = (unsigned int) v_idx + 2;

				v_idx += 3;
			}

			/* Add connecting lines between shapes */
			if (i < count - 1) {
				float nx = (float) weights[i + 1].direction[0] * 3.0f;
				float ny = (float) weights[i + 1].direction[1] * 3.0f;
				float nz = (float) weights[i + 1].direction[2] * 3.0f;

				for (k = 0; k < 4; k++) {
					float t = (float) k / 3.0f;
					vertices[v_idx + k].position[0] = x + (nx - x) * t;
					vertices[v_idx + k].position[1] = y + (ny - y) * t;
					vertices[v_idx + k].position[2] = z + (nz - z) * t;
					vertices[v_idx + k].color[0] = 0.8f;
					vertices[v_idx + k].color[1] = 0.8f;
					vertices[v_idx + k].color[2] = 1.0f;
					vertices[v_idx + k].color[3] = 0.3f;
					vertices[v_idx + k].normal[0] = 0.0f;
					vertices[v_idx + k].normal[1] = 1.0f;
					vertices[v_idx + k].normal[2] = 0.0f;
					vertices[v_idx + k].tex_coord[0] = t;
					vertices[v_idx + k].tex_coord[1] = 0.0f;
				}

				indices[i_idx++] = (unsigned int) v_idx;
				indices[i_idx++] = (unsigned int) v_idx + 1;
				indices[i_idx++] = (unsigned int) v_idx + 2;
				indices[i_idx++] = (unsigned int) v_idx + 2;
				indices[i_idx++] = (unsigned int) v_idx + 3;
				indices[i_idx++] = (unsigned int) v_idx;

				v_idx += 4;
			}
		}

		*vertex_count = v_idx;
		*index_count = i_idx;
	}

	/* ----------------------------------------------------------------------------
	 * Audio Functions
	 * ------------------------------------------------------------------------- */

	int init_audio(AudioEngine *audio) {
		if (!audio)
			return -1;
		memset(audio, 0, sizeof(AudioEngine));

		audio->device = alcOpenDevice(NULL);
		if (!audio->device) {
			log_error("Failed to open audio device");
			return -1;
		}

		audio->context = alcCreateContext(audio->device, NULL);
		if (!audio->context) {
			alcCloseDevice(audio->device);
			log_error("Failed to create audio context");
			return -1;
		}

		alcMakeContextCurrent(audio->context);

		alGenSources(1, &audio->source);
		alGenBuffers(1, &audio->buffer);

		audio->listener_pos[0] = 0.0f;
		audio->listener_pos[1] = 0.0f;
		audio->listener_pos[2] = 0.0f;
		alListenerfv(AL_POSITION, audio->listener_pos);

		audio->listener_ori[0] = 0.0f;
		audio->listener_ori[1] = 0.0f;
		audio->listener_ori[2] = -1.0f;
		audio->listener_ori[3] = 0.0f;
		audio->listener_ori[4] = 1.0f;
		audio->listener_ori[5] = 0.0f;
		alListenerfv(AL_ORIENTATION, audio->listener_ori);

		audio->base_frequency = 220.0;
		audio->frequency_range = 220.0;
		audio->initialized = 1;

		return 0;
	}

	void cleanup_audio(AudioEngine *audio) {
		if (!audio || !audio->initialized)
			return;

		if (audio->source) {
			alSourceStop(audio->source);
			alDeleteSources(1, &audio->source);
		}
		if (audio->buffer)
			alDeleteBuffers(1, &audio->buffer);
		if (audio->context) {
			alcMakeContextCurrent(NULL);
			alcDestroyContext(audio->context);
		}
		if (audio->device)
			alcCloseDevice(audio->device);

		memset(audio, 0, sizeof(AudioEngine));
	}

	void play_spatial_audio(AudioEngine *audio, double frequency, double pan,
			double volume) {
		ALshort data[2048];
		int i;
		double phase = 0.0;
		double phase_inc;

		if (!audio || !audio->initialized)
			return;

		phase_inc = 2.0 * 3.14159 * frequency / 44100.0;

		for (i = 0; i < 2048; i++) {
			/* Create a richer sound with harmonics */
			double sample = sin(phase) * 0.5;
			sample += sin(phase * 2.0) * 0.25;
			sample += sin(phase * 3.0) * 0.125;
			sample += sin(phase * 4.0) * 0.0625;
			data[i] = (ALshort) (sample * 32767.0 * volume);
			phase += phase_inc;
			if (phase > 2.0 * 3.14159)
				phase -= 2.0 * 3.14159;
		}

		alBufferData(audio->buffer, AL_FORMAT_MONO16, data,
				2048 * sizeof(ALshort), 44100);
		alSource3f(audio->source, AL_POSITION, (float) pan * 5.0f, 0.0f, 2.0f);
		alSourcei(audio->source, AL_BUFFER, audio->buffer);
		alSourcePlay(audio->source);
	}

	/* ----------------------------------------------------------------------------
	 * Window Management
	 * ------------------------------------------------------------------------- */

	int init_window(WindowManager *wm, int width, int height, const char *title) {
		if (!wm)
			return -1;
		memset(wm, 0, sizeof(WindowManager));

		if (SDL_Init(SDL_INIT_VIDEO) < 0) {
			log_error("SDL initialization failed: %s", SDL_GetError());
			return -1;
		}

		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
				SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

		wm->window = SDL_CreateWindow(title,
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, width, height,
				SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

		if (!wm->window) {
			log_error("Window creation failed: %s", SDL_GetError());
			SDL_Quit();
			return -1;
		}

		wm->gl_context = SDL_GL_CreateContext(wm->window);
		if (!wm->gl_context) {
			log_error("OpenGL context creation failed: %s", SDL_GetError());
			SDL_DestroyWindow(wm->window);
			SDL_Quit();
			return -1;
		}

		SDL_GL_SetSwapInterval(1);

		wm->width = width;
		wm->height = height;
		wm->camera_distance = 8.0f;
		wm->camera_rotation = 0.0f;
		wm->quit_requested = 0;
		wm->input_length = 0;
		memset(wm->keyboard_state, 0, sizeof(wm->keyboard_state));
		memset(wm->input_buffer, 0, sizeof(wm->input_buffer));

		return 0;
	}

	void cleanup_window(WindowManager *wm) {
		if (!wm)
			return;

		if (wm->gl_context)
			SDL_GL_DeleteContext(wm->gl_context);
		if (wm->window)
			SDL_DestroyWindow(wm->window);
		SDL_Quit();
		memset(wm, 0, sizeof(WindowManager));
	}

	int process_events(WindowManager *wm) {
		SDL_Event event;

		if (!wm)
			return -1;

		while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_QUIT:
				wm->quit_requested = 1;
				break;

			case SDL_KEYDOWN:
				if (event.key.keysym.scancode < SDL_NUM_SCANCODES) {
					wm->keyboard_state[event.key.keysym.scancode] = 1;
				}
				handle_keyboard_input(wm, event.key.keysym.sym);
				break;

			case SDL_KEYUP:
				if (event.key.keysym.scancode < SDL_NUM_SCANCODES) {
					wm->keyboard_state[event.key.keysym.scancode] = 0;
				}
				break;

			case SDL_MOUSEMOTION:
				wm->mouse_x = event.motion.x;
				wm->mouse_y = event.motion.y;
				break;

			case SDL_MOUSEBUTTONDOWN:
				wm->mouse_buttons |= (1 << event.button.button);
				break;

			case SDL_MOUSEBUTTONUP:
				wm->mouse_buttons &= ~(1 << event.button.button);
				break;

			case SDL_WINDOWEVENT:
				if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					wm->width = event.window.data1;
					wm->height = event.window.data2;
					if (glViewport_ptr) {
						glViewport_ptr(0, 0, wm->width, wm->height);
					}
					update_projection_matrix(g_state.gl.projection, 45.0f,
							(float) wm->width / (float) wm->height, 0.1f,
							100.0f);
				}
				break;

			default:
				break;
			}
		}

		/* Camera controls */
		if (wm->keyboard_state[SDL_SCANCODE_UP]
				|| wm->keyboard_state[SDL_SCANCODE_W]) {
			wm->camera_distance -= 0.1f;
			if (wm->camera_distance < 3.0f)
				wm->camera_distance = 3.0f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_DOWN]
				|| wm->keyboard_state[SDL_SCANCODE_S]) {
			wm->camera_distance += 0.1f;
			if (wm->camera_distance > 20.0f)
				wm->camera_distance = 20.0f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_LEFT]
				|| wm->keyboard_state[SDL_SCANCODE_A]) {
			wm->camera_rotation += 0.02f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_RIGHT]
				|| wm->keyboard_state[SDL_SCANCODE_D]) {
			wm->camera_rotation -= 0.02f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_SPACE]) {
			wm->camera_distance = 8.0f;
			wm->camera_rotation = 0.0f;
		}

		return wm->quit_requested ? 1 : 0;
	}

	void handle_keyboard_input(WindowManager *wm, SDL_Keycode key) {
		if (!wm)
			return;

		switch (key) {
		case SDLK_ESCAPE:
			wm->quit_requested = 1;
			break;

		case SDLK_RETURN:
			if (wm->input_length > 0) {
				log_info("Processing: %s", wm->input_buffer);
				wm->input_length = 0;
			}
			break;

		case SDLK_BACKSPACE:
			if (wm->input_length > 0) {
				wm->input_buffer[--wm->input_length] = '\0';
			}
			break;

		default:
			if (key >= 32 && key <= 126
					&& wm->input_length < sizeof(wm->input_buffer) - 1) {
				wm->input_buffer[wm->input_length++] = (char) key;
			}
			break;
		}
	}

	/* ----------------------------------------------------------------------------
	 * Core Pipeline Functions
	 * ------------------------------------------------------------------------- */

	int tokenize_text(const char *text, TokenFuzzyState *tokens,
			size_t max_tokens) {
		size_t i;
		unsigned int hash = 5381;
		int c;

		if (!text || !tokens || max_tokens == 0)
			return -1;

		const char *word_start = text;
		const char *word_end;
		size_t token_idx = 0;

		while (*word_start && token_idx < max_tokens) {
			/* Skip whitespace */
			while (*word_start
					&& (*word_start == ' ' || *word_start == '\t'
							|| *word_start == '\n' || *word_start == '\r')) {
				word_start++;
			}

			if (!*word_start)
				break;

			/* Find word end */
			word_end = word_start;
			while (*word_end && *word_end != ' ' && *word_end != '\t'
					&& *word_end != '\n' && *word_end != '\r') {
				word_end++;
			}

			/* Compute hash for token */
			hash = 5381;
			for (c = *word_start; word_start < word_end; word_start++) {
				hash = ((hash << 5) + hash) + c;
			}

			TokenFuzzyState *token = &tokens[token_idx];

			for (i = 0; i < MAX_DIMENSIONS; i++) {
				unsigned int byte = (hash >> (i * 8)) & 0xFF;
				token->membership_values[i] = (double) byte / 255.0;
				token->membership_values[i] = token->membership_values[i]
						* token->membership_values[i];
			}

			for (i = 0; i < MAX_DIMENSIONS; i++) {
				double m = token->membership_values[i];
				double certainty = 4.0 * m * (1.0 - m);
				token->entropy_weights[i] = 1.0 - certainty;
			}

			token->token_count = 1;
			token->dimension = MAX_DIMENSIONS;
			token->sequence_id = (unsigned short) token_idx;
			token->confidence = 0.5;
			token->timestamp = get_timestamp();

			token_idx++;
		}

		return (int) token_idx;
	}

	void fuzzy_inference(TokenFuzzyState *tokens, const MandaniRule *rules,
			size_t num_tokens, size_t num_rules) {
		size_t i, j, k;

		if (!tokens || !rules || num_tokens == 0 || num_rules == 0)
			return;

		for (i = 0; i < num_tokens; i++) {
			TokenFuzzyState *token = &tokens[i];
			double rule_activation[MAX_DIMENSIONS] = { 0 };
			double rule_count[MAX_DIMENSIONS] = { 0 };

			for (j = 0; j < num_rules; j++) {
				const MandaniRule *rule = &rules[j];
				double rule_strength = 1.0;

				for (k = 0; k < MAX_DIMENSIONS; k++) {
					rule_strength *= rule->antecedents[k];
				}

				if (rule_strength > RULE_STRENGTH_MIN) {
					for (k = 0; k < MAX_DIMENSIONS; k++) {
						rule_activation[k] += rule->consequents[k]
								* rule_strength;
						rule_count[k] += rule_strength;
					}
				}
			}

			for (k = 0; k < MAX_DIMENSIONS; k++) {
				if (rule_count[k] > 0) {
					token->membership_values[k] = rule_activation[k]
							/ rule_count[k];
				}
			}

			for (k = 0; k < MAX_DIMENSIONS; k++) {
				double m = token->membership_values[k];
				double entropy = -m * log(m + MEMBERSHIP_EPSILON)
						- (1 - m) * log(1 - m + MEMBERSHIP_EPSILON);
				token->entropy_weights[k] = entropy / log(2.0);
			}
		}
	}

	void entropy_optimization(TokenFuzzyState *tokens, WeightVector *weights,
			double target_entropy) {
		size_t i;

		if (!tokens || !weights)
			return;
		(void) weights; /* Suppress unused warning */

		for (i = 0; i < MAX_TOKENS; i++) {
			if (tokens[i].token_count > 0) {
				double token_entropy = compute_entropy(
						tokens[i].entropy_weights,
						MAX_DIMENSIONS);

				if (fabs(token_entropy - target_entropy) > ENTROPY_THRESHOLD) {
					double scale = target_entropy
							/ (token_entropy + MEMBERSHIP_EPSILON);
					int j;

					for (j = 0; j < MAX_DIMENSIONS; j++) {
						tokens[i].entropy_weights[j] *= scale;
						if (tokens[i].entropy_weights[j] > 1.0) {
							tokens[i].entropy_weights[j] = 1.0;
						}
					}
				}
			}
		}
	}

	void vector_compute(WeightVector *weights, const TokenFuzzyState *tokens,
			size_t count) {
		size_t i, j;

		if (!weights || !tokens || count == 0)
			return;

		for (i = 0; i < count; i++) {
			const TokenFuzzyState *token = &tokens[i];
			WeightVector *vec = &weights[i];

			if (token->token_count == 0)
				continue;

			for (j = 0; j < MAX_DIMENSIONS; j++) {
				vec->axis_weights[j] = token->membership_values[j];
				vec->dimension_factors[j] = token->entropy_weights[j];
				vec->combined_vector[j] = token->membership_values[j]
						* token->entropy_weights[j];
			}

			double mag_sq = 0.0;
			for (j = 0; j < MAX_DIMENSIONS; j++) {
				mag_sq += vec->combined_vector[j] * vec->combined_vector[j];
			}
			vec->magnitude = sqrt(mag_sq);

			if (vec->magnitude > MEMBERSHIP_EPSILON) {
				double inv_mag = 1.0 / vec->magnitude;
				for (j = 0; j < MAX_DIMENSIONS; j++) {
					vec->direction[j] = vec->combined_vector[j] * inv_mag;
				}
			}

			vec->flags = (token->token_count << 16) | (unsigned int) i;
		}
	}

	/* ----------------------------------------------------------------------------
	 * System Initialization
	 * ------------------------------------------------------------------------- */

	int initialize_system(NeuroFuzzySystem **system) {
		if (!system)
			return -1;

		*system = (NeuroFuzzySystem*) aligned_alloc_wrapper(CACHE_LINE_SIZE,
				sizeof(NeuroFuzzySystem));
		if (!*system) {
			log_error("Failed to allocate neuro-fuzzy system");
			return -1;
		}

		memset(*system, 0, sizeof(NeuroFuzzySystem));

		(*system)->token_states = (TokenFuzzyState*) aligned_alloc_wrapper(
		SIMD_ALIGNMENT, MAX_TOKENS * sizeof(TokenFuzzyState));
		if (!(*system)->token_states) {
			log_error("Failed to allocate token states");
			aligned_free_wrapper(*system);
			return -1;
		}
		memset((*system)->token_states, 0,
				MAX_TOKENS * sizeof(TokenFuzzyState));

		(*system)->rule_base = (MandaniRule*) aligned_alloc_wrapper(
		SIMD_ALIGNMENT, MAX_RULES * sizeof(MandaniRule));
		if (!(*system)->rule_base) {
			log_error("Failed to allocate rule base");
			aligned_free_wrapper((*system)->token_states);
			aligned_free_wrapper(*system);
			return -1;
		}
		memset((*system)->rule_base, 0, MAX_RULES * sizeof(MandaniRule));

		(*system)->current_weights = (WeightVector*) aligned_alloc_wrapper(
		SIMD_ALIGNMENT, MAX_TOKENS * sizeof(WeightVector));
		if (!(*system)->current_weights) {
			log_error("Failed to allocate weight vectors");
			aligned_free_wrapper((*system)->rule_base);
			aligned_free_wrapper((*system)->token_states);
			aligned_free_wrapper(*system);
			return -1;
		}
		memset((*system)->current_weights, 0,
				MAX_TOKENS * sizeof(WeightVector));

		/* Initialize some example rules */
		int r;
		for (r = 0; r < 8; r++) {
			MandaniRule *rule = &(*system)->rule_base[r];
			int d;
			for (d = 0; d < MAX_DIMENSIONS; d++) {
				rule->antecedents[d] = (double) ((r >> d) & 1);
				rule->consequents[d] = (double) ((r >> (MAX_DIMENSIONS - 1 - d))
						& 1);
			}
			rule->rule_strength = 0.5 + (double) r / 16.0;
			rule->confidence = 0.8;
			rule->antecedent_count = MAX_DIMENSIONS;
			rule->consequent_count = MAX_DIMENSIONS;
		}
		(*system)->rule_count = 8;

		(*system)->token_count = 0;
		(*system)->global_entropy = 0.5;
		(*system)->processing_dimensions = MAX_DIMENSIONS;
		(*system)->frame_counter = 0;

		clock_gettime(CLOCK_MONOTONIC, &(*system)->last_update);

		return 0;
	}

	void cleanup_system(NeuroFuzzySystem *system) {
		if (!system)
			return;

		if (system->token_states)
			aligned_free_wrapper(system->token_states);
		if (system->rule_base)
			aligned_free_wrapper(system->rule_base);
		if (system->current_weights)
			aligned_free_wrapper(system->current_weights);
		aligned_free_wrapper(system);
	}

	int initialize_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
			AudioEngine *audio, WindowManager *wm) {
		int ret = 0;

		log_info("Initializing EVOX Neuro-Fuzzy Visualization Engine v%s",
				PROJECT_VERSION);
		log_info("System: %s",
				HAVE_AVX2 ? "AVX2/FMA optimized" : "Scalar mode");

		if (numa_available() >= 0) {
			log_info("NUMA available with %d nodes",
					numa_num_configured_nodes());
		} else {
			log_info("NUMA not available, using single node");
		}

		log_info("Initializing OpenCL 3.0...");
		ret = init_opencl(ocl);
		if (ret != 0) {
			log_error(
					"OpenCL initialization failed, continuing without GPU acceleration");
		} else {
			log_info("OpenCL initialized successfully");
		}

		log_info("Creating window (%dx%d)...", WINDOW_WIDTH, WINDOW_HEIGHT);
		ret = init_window(wm, WINDOW_WIDTH, WINDOW_HEIGHT, PROJECT_NAME);
		if (ret != 0) {
			log_error("Window initialization failed");
			cleanup_opencl(ocl);
			return -1;
		}

		log_info("Initializing OpenGL 3.3...");
		ret = init_opengl(gl, WINDOW_WIDTH, WINDOW_HEIGHT);
		if (ret != 0) {
			log_error("OpenGL initialization failed");
			cleanup_window(wm);
			cleanup_opencl(ocl);
			return -1;
		}

		log_info("Initializing OpenAL...");
		ret = init_audio(audio);
		if (ret != 0) {
			log_error("Audio initialization failed, continuing without audio");
		} else {
			log_info("Audio initialized successfully");
		}

		log_info("All subsystems initialized successfully");
		log_info("Type text and press Enter to generate visualization");
		log_info(
				"Controls: Arrow keys/WASD to move camera, Space to reset, ESC to exit");

		return 0;
	}

	void cleanup_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
			AudioEngine *audio, WindowManager *wm) {
		log_info("Cleaning up subsystems...");

		cleanup_audio(audio);
		cleanup_opengl(gl);
		cleanup_window(wm);
		cleanup_opencl(ocl);

		log_info("Cleanup complete");
	}

	/* ----------------------------------------------------------------------------
	 * Main Loop
	 * ------------------------------------------------------------------------- */

	int main_loop(NeuroFuzzySystem *system, OpenCLWrapper *ocl,
			OpenGLRenderer *gl, AudioEngine *audio, WindowManager *wm) {
		struct timespec frame_start, frame_end;
		long frame_time_us;
		double target_entropy = 0.5;
		static RenderVertex vertices[MAX_VERTICES * 2];
		static unsigned int indices[MAX_VERTICES * 3];
		size_t vertex_count = 0;
		size_t index_count = 0;
		int ret;
		float time = 0.0f;
		int last_processed = 0;

		(void) ocl;

		while (!wm->quit_requested) {
			clock_gettime(CLOCK_MONOTONIC, &frame_start);
			time += 0.016f; /* Approximately 60 FPS increment */

			/* Process input events */
			ret = process_events(wm);
			if (ret != 0)
				break;

			/* Update view matrix based on camera */
			update_view_matrix(gl->view, wm->camera_distance,
					wm->camera_rotation);

			/* Check for input text */
			if (wm->input_length > 0
					&& wm->input_length != (size_t) last_processed) {
				int num_tokens = tokenize_text(wm->input_buffer,
						system->token_states, MAX_TOKENS);
				if (num_tokens > 0) {
					system->token_count = (size_t) num_tokens;
					log_info("Tokenized %zu tokens from: '%s'",
							system->token_count, wm->input_buffer);

					fuzzy_inference(system->token_states, system->rule_base,
							system->token_count, system->rule_count);

					entropy_optimization(system->token_states,
							system->current_weights, target_entropy);

					vector_compute(system->current_weights,
							system->token_states, system->token_count);

					last_processed = (int) wm->input_length;
				}
			}

			/* Generate rich visualization */
			if (system->token_count > 0) {
				generate_rich_visualization(system->current_weights,
						system->token_count, vertices, indices, &vertex_count,
						&index_count, time);
			} else {
				/* Generate a default rotating shape when no input */
				WeightVector default_weights[1];
				memset(default_weights, 0, sizeof(default_weights));
				default_weights[0].magnitude = 1.0f;
				default_weights[0].direction[0] = sinf(time * 0.5f);
				default_weights[0].direction[1] = cosf(time * 0.3f);
				default_weights[0].direction[2] = sinf(time * 0.7f);
				default_weights[0].flags = ((unsigned int) (time * 100) & 0xFF)
						<< 16;

				generate_rich_visualization(default_weights, 1, vertices,
						indices, &vertex_count, &index_count, time);
			}

			/* Render frame */
			render_cad_wireframe(gl, vertices, vertex_count, indices,
					index_count, time);

			/* Swap buffers */
			SDL_GL_SwapWindow(wm->window);

			/* Play audio based on system entropy */
			if (audio->initialized && system->token_count > 0) {
				system->global_entropy = compute_entropy(
						(double*) system->current_weights,
						(int) (system->token_count * MAX_DIMENSIONS));

				double frequency = audio->base_frequency
						+ system->global_entropy * audio->frequency_range;
				double pan = (system->global_entropy - 0.5) * 2.0;
				double volume = 0.3 + system->global_entropy * 0.5;
				play_spatial_audio(audio, frequency, pan, volume);
			}

			/* Frame rate limiting */
			clock_gettime(CLOCK_MONOTONIC, &frame_end);
			frame_time_us = (frame_end.tv_sec - frame_start.tv_sec) * 1000000
					+ (frame_end.tv_nsec - frame_start.tv_nsec) / 1000;

			if (frame_time_us < FRAME_TIME_US) {
				sleep_us(FRAME_TIME_US - frame_time_us);
			}

			system->frame_counter++;

			/* Print FPS occasionally */
			if (system->frame_counter % 300 == 0) {
				log_info("FPS: %.1f, Tokens: %zu, Entropy: %.3f",
						1000000.0f / (float) frame_time_us, system->token_count,
						system->global_entropy);
			}
		}

		return 0;
	}

	/* ----------------------------------------------------------------------------
	 * Entry Point
	 * ------------------------------------------------------------------------- */

	int main(int argc, char **argv) {
		int ret = 0;
		pthread_barrier_t numa_barrier;
		NUMAThreadContext *thread_contexts = NULL;

		(void) argc;
		(void) argv;

		log_info("%s v%s starting...", PROJECT_NAME, PROJECT_VERSION);

		/* Initialize main system */
		ret = initialize_system(&g_state.system);
		if (ret != 0) {
			log_error("Failed to initialize system");
			return EXIT_FAILURE;
		}

		/* Initialize all subsystems */
		ret = initialize_subsystems(&g_state.ocl, &g_state.gl, &g_state.audio,
				&g_state.wm);
		if (ret != 0) {
			log_error("Failed to initialize subsystems");
			cleanup_system(g_state.system);
			return EXIT_FAILURE;
		}

		/* Initialize barrier for NUMA threads */
		pthread_barrier_init(&numa_barrier, NULL, 2);

		/* Create NUMA-aware threads for processing */
		int num_cores = get_nprocs_conf();
		ret = create_numa_threads(&thread_contexts, num_cores, g_state.system,
				&numa_barrier);
		if (ret != 0) {
			log_info("Running in single-threaded mode");
		}

		/* Main processing loop */
		g_state.running = 1;
		ret = main_loop(g_state.system, &g_state.ocl, &g_state.gl,
				&g_state.audio, &g_state.wm);

		/* Cleanup threads */
		if (thread_contexts) {
			join_numa_threads(&thread_contexts, num_cores);
		}
		pthread_barrier_destroy(&numa_barrier);

		/* Cleanup */
		cleanup_subsystems(&g_state.ocl, &g_state.gl, &g_state.audio,
				&g_state.wm);
		cleanup_system(g_state.system);

		log_info("EVOX engine terminated normally");
		return EXIT_SUCCESS;
	}

	/* ============================================================================
	 * End of File
	 * ============================================================================ */
