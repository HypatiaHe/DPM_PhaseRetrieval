/*

 _____  _____  __  __
 |  __ \|  __ \|  \/  |
 | |  | | |__) | \  / |
 | |  | |  ___/| |\/| |
 | |__| | |    | |  | |
 |_____/|_|    |_|  |_|

 -> Typical Diffraction Phase Microsocpy (DPM) image reconstruction steps
 --->Phase retrieval from fringes using a background followed by fourier filtering
 ---->Typical performance is >20ms on desktop cards for 2MP input

 Version History:
 1. Pulled things out from SLIM control software, tested with MSVC2013 against CUDA 8 on 770M

 ->Mikhail Kandel 9/26
 -->kandel3@Illinois.edu

 */
#include <tiffio.h>
#include <chrono>
#include <string>
#include <algorithm>
#include <map>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <sstream>

size_t timestamp()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

template<typename T> static void ThrustSafeResize(thrust::device_vector<T>& in, size_t numel)
{

	if (in.size() != numel)
	{
		// This check didn't work correctly on old implementations of thrust
		try
		{
			in.resize(numel);
		}
		catch (const thrust::system_error& e)
		{
			std::cerr << e.what() << std::endl;
			throw;
		}
	}
}

template<typename T> static T* ThrustSafeGetPointer(thrust::device_vector<T>& in, size_t numel, bool check = false)
{
	if (numel == 0)
	{
		throw std::runtime_error("Requested an empty buffer");
	}
	if ((in.size() != numel) && check)
	{
		throw std::runtime_error("Performance problem, should be pre_allocated");
	}
	ThrustSafeResize(in, numel);
	return thrust::raw_pointer_cast(in.data());
}

#define CudaSafeCall( err ) __CudaSafeCall(err, __FILE__, __LINE__ )
inline void __CudaSafeCall(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		std::stringstream error_msg;
		error_msg << "CudaSafeCall() failed at " << file << ":" << line << ":" << cudaGetErrorString(err);
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
}

#define CufftSafeCall( err ) __CufftSafeCallCall( err, __FILE__, __LINE__ )
static const char *_cufftGetErrorEnum(cufftResult_t error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";
	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";
	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";
	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";
	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";
	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";
	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";
	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";
	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	case CUFFT_INCOMPLETE_PARAMETER_LIST: break;
	case CUFFT_INVALID_DEVICE: break;
	case CUFFT_PARSE_ERROR: break;
	case CUFFT_NO_WORKSPACE: break;
	case CUFFT_NOT_IMPLEMENTED: break;
	case CUFFT_LICENSE_ERROR: break;
	default: break;
	}
	return "<unknown>";
}

inline void __CufftSafeCallCall(cufftResult_t err, const char *file, const int line)
{
	if (CUFFT_SUCCESS != err)
	{
		std::stringstream error_msg;
		error_msg << "CufftSafeCall() failed at " << file << ":" << line << ":" << _cufftGetErrorEnum(err);
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
}
typedef std::pair<int, bool> CufftPlanWrapper;
void CufftPlanDealloc(const CufftPlanWrapper& in)
{
	if (in.second)
	{
		CufftSafeCall(cufftDestroy(in.first));//make check cufft_result?
	}
}
template<typename T>
static void WriteTiff(const std::string& name, const T* ptr, unsigned int cols, unsigned int rows)
{
	auto tif = TIFFOpen(name.c_str(), "w");//quality error correcting code
	if (tif == nullptr)
	{
		std::stringstream error_msg;
		error_msg << "Can't Open File " << name;
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, cols);  // set the width of the image
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, rows);    // set the height of the image
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
	TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
	TIFFSetField(tif, TIFFTAG_MODEL, __DATE__);
	TIFFSetField(tif, TIFFTAG_MAKE, "Mikhail Kandel");
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8 * sizeof(T));
	auto rowsize = cols*sizeof(T);
	// c++ needs a static_if http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3329.pdf
	if (std::is_same<T, float>::value || std::is_same<T, double>::value)
	{
		TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
	}
	else
	{
		TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
	}
	///
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
	for (unsigned int r = 0; r < rows; r++)
	{
		auto aschar = (unsigned char*)(ptr);//need to get rid of const due to problems with libtiff
		auto data = &aschar[rowsize*r];
		TIFFWriteScanline(tif, data, r);
	}
	TIFFClose(tif);
}

struct FrameSize
{
	// Note this fails for images larger than 2^31
	int width, height;
	FrameSize() : FrameSize(0, 0){}
	explicit FrameSize(int Width, int Height) : width(Width), height(Height){ ; }
	int n() const
	{
		return width*height;
	}
	bool operator== (const FrameSize &c1) const
	{
		return (width == c1.width && height == c1.height);
	}
	bool operator!= (const FrameSize &c1) const
	{
		return !(*this == c1);
	}
	bool operator<(const FrameSize& rhs) const
	{
		return n() < rhs.n();
	}
};

struct Image : FrameSize
{
	thrust::host_vector<unsigned short> img;
};

static Image ReadBuffer(const std::string& name)
{
	auto tif = TIFFOpen(name.c_str(), "r");
	if (tif == nullptr)
	{
		std::stringstream error_msg;
		error_msg << "Can't Open File " << name;
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
	int imgH, imgW;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgH);
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgW);
	unsigned short bits_per_sample;
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
	if (bits_per_sample != sizeof(unsigned short) * 8)
	{
		std::stringstream error_msg;
		error_msg << "File: " + std::string(name) + " has an unssuported bit depth of " + std::to_string(bits_per_sample);
		auto error_msg_str = error_msg.str();
		std::cout << error_msg_str << std::endl;
		throw std::runtime_error(error_msg_str);
	}
	Image ret;
	int rowsize = imgW*sizeof(unsigned short);
	ret.img.resize(imgW*imgH);
	static_cast<FrameSize&>(ret) = FrameSize(imgW, imgH);
	auto mydata = reinterpret_cast<unsigned char*>(ret.img.data());
	for (auto row = 0; row < imgH; row++)
	{
		auto toMe = static_cast<void*>(&mydata[rowsize*row]);
		TIFFReadScanline(tif, toMe, row);
	}
	TIFFClose(tif);
	return ret;
}

#ifdef _DEBUG
#define debug_writes true
#else
#define debug_writes false
#endif

template<typename T>
void WriteDebugCuda(const T* pointer_in, const size_t width, const size_t height, const char* name, bool do_it_anyways = false)
{
	if (do_it_anyways || debug_writes)
	{
		//todo replace with thrust?
		auto bytes = width*height*sizeof(T);
		T* temp = nullptr;
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaHostAlloc((void**)&temp, bytes, cudaHostAllocDefault));
		CudaSafeCall(cudaMemcpy(temp, pointer_in, bytes, cudaMemcpyDeviceToHost));
		WriteTiff(name, temp, width, height);
		CudaSafeCall(cudaFreeHost(temp));
	}
}

void WriteDebugCuda(const cufftComplex* pointer_in, const size_t width, const size_t height, const char* name, bool do_it_anyways = false)
{
	if (do_it_anyways || debug_writes)
	{
		//todo replace with thrust?
		CudaSafeCall(cudaDeviceSynchronize());
		auto numel = width*height;
		auto bytes = numel*sizeof(cufftComplex);
		auto bytesR = numel*sizeof(cufftReal);
		cufftComplex* tempc;
		CudaSafeCall(cudaHostAlloc((void**)&tempc, bytes, cudaHostAllocDefault));
		CudaSafeCall(cudaMemcpy(tempc, pointer_in, bytes, cudaMemcpyDeviceToHost));
		cufftReal* temp;
		CudaSafeCall(cudaHostAlloc((void**)&temp, bytesR, cudaHostAllocDefault));
		for (auto i = 0; i < numel; i++)
		{
			auto v = tempc[i];
			temp[i] = sqrtf(v.x*v.x + v.y*v.y);
		}
		WriteTiff(name, temp, width, height);
		CudaSafeCall(cudaFreeHost(tempc));
		CudaSafeCall(cudaFreeHost(temp));
	}
}

struct BandPassSettings
{
	bool do_band_pass, remove_dc;
	float min_dx, max_dx;
	float min_dy, max_dy;
	BandPassSettings(float MinDx, float MaxDx, float MinDy, float MaxDy, bool Remove_dc, bool DoBandPass) : do_band_pass(DoBandPass), remove_dc(Remove_dc), min_dx(MinDx), max_dx(MaxDx), min_dy(MinDy), max_dy(MaxDy){};
	BandPassSettings() :BandPassSettings(0, 0, 0, 0, false, false){};
	friend bool operator== (const BandPassSettings &a, const BandPassSettings &b)
	{
		return (a.do_band_pass == b.do_band_pass) && (a.min_dx == b.min_dx) && (a.max_dx == b.max_dx) && (a.min_dy == b.min_dy) && (a.max_dy == b.max_dy) && (a.remove_dc == b.remove_dc);
	}
	friend bool operator!= (const BandPassSettings &a, const BandPassSettings &b)
	{
		return !(a == b);
	}
};

__global__ void _LoadImage(cuComplex* dst, const float* src, int w_in, int h_in, int w_out, int h_out)
{
	//There might be an NPP cof this
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
	{
		auto c_pad = (w_out - w_in) / 2;
		auto r_pad = (h_out - h_in) / 2;
		//
		auto r_new = r - r_pad;//todo this can be optomized
		auto r_box = r_new / h_in;
		r_new = abs(r_new % h_in);
		if (!(r_box % 2 == 0))
		{
			r_new = h_in - r_new - 1;
		}
		//
		auto c_new = c - c_pad;//todo this can be optomized
		auto c_box = c_new / w_in;
		c_new = abs(c_new % w_in);
		if (!(c_box % 2 == 0))
		{
			c_new = w_in - c_new - 1;
		}
		//
		auto in_idx = r_new*w_in + c_new;
		auto out_idx = r*w_out + c;//I think there was a typo here...
		//
		dst[out_idx].x = src[in_idx];
		dst[out_idx].y = 0;
	}
}

struct ScaleMagnitude
{
	//  tell  CUDA that the following code can be executed on the CPU and the GPU
	__host__ __device__  cuComplex  operator()(const cuComplex& x, const float& y) const
	{
		return{ x.x*y, x.y*y };
	}
};

void loadImage(thrust::device_vector<cuComplex>& img_big, const float* src_ptr, const FrameSize& frame_in, const FrameSize& frame_out)
{
	auto dst_ptr = ThrustSafeGetPointer(img_big, frame_out.n());
	//
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	gs2d.x = static_cast<unsigned int>(ceil(frame_out.width / (1.f* bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(frame_out.height / (1.f* bs2d.y)));
	_LoadImage << <gs2d, bs2d >> >(dst_ptr, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);
}

__global__ void _GetBackImage(float* dst, const cuComplex* src, int w_in, int h_in, int w_out, int h_out)
{
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
	{
		auto c_pad = (w_in - w_out) / 2;
		auto r_pad = (h_in - h_out) / 2;
		auto r_new = r_pad + r;
		auto c_new = c_pad + c;
		//
		if ((r_new < h_in) && (c_new < w_in))//dont think this can happen..
		{
			auto in_idx = r_new*w_in + c_new;
			auto out_idx = r*w_out + c;
			//
			auto val = src[in_idx];
			dst[out_idx] = val.x;//This scaling is done at the output / (w_in*w_out);
		}
	}
}

__global__ void _GetBackImage_Imag(float* dst, const cuComplex* src, int w_in, int h_in, int w_out, int h_out)
{
	int c = threadIdx.x + blockIdx.x*blockDim.x;
	int r = threadIdx.y + blockIdx.y*blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
	{
		auto c_pad = (w_in - w_out) / 2;
		auto r_pad = (h_in - h_out) / 2;
		auto r_new = r_pad + r;
		auto c_new = c_pad + c;
		//
		if ((r_new < h_in) && (c_new < w_in))//dont think this can happen..
		{
			auto in_idx = r_new*w_in + c_new;
			auto out_idx = r*w_out + c;
			//
			auto val = src[in_idx];
			dst[out_idx] = val.y;//This scaling is done at the output / (w_in*w_out);
		}
	}
}

void GetBackImage(cufftReal* img, bool real_part, thrust::device_vector<cuComplex>& img_big, const FrameSize& frame_in, const FrameSize& frame_out)
{
	auto  src_ptr = thrust::raw_pointer_cast(img_big.data());
	//
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	gs2d.x = static_cast<unsigned int>(ceil(frame_out.width / (1.f* bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(frame_out.height / (1.f* bs2d.y)));
	if (real_part)
	{
		_GetBackImage << <gs2d, bs2d >> >(img, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);
	}
	else
	{
		_GetBackImage_Imag << <gs2d, bs2d >> >(img, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);//An alternative is multiply by i or something
	}
}

class ScaleByConstant
{
	//http://stackoverflow.com/questions/14441142/scaling-in-inverse-fft-by-cufft
private:
	float c_;

public:
	explicit ScaleByConstant(float c) { c_ = c; };

	__host__ __device__ float operator()(float &a) const
	{
		auto output = a * c_;
		return output;
	}

};

class FourierFilter
{
	CufftPlanWrapper ft_plan;
	int expanded_width;
	thrust::device_vector<cuComplex> img_expanded;
	thrust::device_vector<cuComplex> img_expanded_ft;
	thrust::device_vector<float> filter_d;
	thrust::host_vector<float> filter_h;
	BandPassSettings old_band;
	FrameSize old_size;
	static void GenerateBandFilter(thrust::host_vector<float>& filter, const BandPassSettings& band, const FrameSize& frame)
	{
		//From https://imagej.nih.gov/ij/plugins/fft-filter.html
		if (band.do_band_pass == false)
		{
			return;
		}
		if (frame.width != frame.height)
		{
			throw std::runtime_error("Frame height and width should be the same");
		}
		auto maxN = static_cast<int>(std::max(frame.width, frame.height));//todo make sure they are the same

		auto filterLargeC = 2.0f*band.max_dx / maxN;
		auto filterSmallC = 2.0f*band.min_dx / maxN;
		auto scaleLargeC = filterLargeC*filterLargeC;
		auto scaleSmallC = filterSmallC*filterSmallC;

		auto filterLargeR = 2.0f*band.max_dy / maxN;
		auto filterSmallR = 2.0f*band.min_dy / maxN;
		auto scaleLargeR = filterLargeR*filterLargeR;
		auto scaleSmallR = filterSmallR*filterSmallR;

		// loop over rows
		for (auto j = 1; j < maxN / 2; j++)
		{
			auto row = j * maxN;
			auto backrow = (maxN - j)*maxN;
			auto rowFactLarge = exp(-(j*j) * scaleLargeR);
			auto rowFactSmall = exp(-(j*j) * scaleSmallR);
			// loop over columns
			for (auto col = 1; col < maxN / 2; col++)
			{
				auto backcol = maxN - col;
				auto colFactLarge = exp(-(col*col) * scaleLargeC);
				auto colFactSmall = exp(-(col*col) * scaleSmallC);
				auto factor = (((1 - rowFactLarge*colFactLarge) * rowFactSmall*colFactSmall));
				filter[col + row] *= factor;
				filter[col + backrow] *= factor;
				filter[backcol + row] *= factor;
				filter[backcol + backrow] *= factor;
			}
		}
		auto fixy = [&](float t){return isinf(t) ? 0 : t; };
		auto rowmid = maxN * (maxN / 2);
		auto rowFactLarge = fixy(exp(-(maxN / 2)*(maxN / 2) * scaleLargeR));
		auto rowFactSmall = fixy(exp(-(maxN / 2)*(maxN / 2) *scaleSmallR));
		filter[maxN / 2] *= ((1 - rowFactLarge) * rowFactSmall);
		filter[rowmid] *= ((1 - rowFactLarge) * rowFactSmall);
		filter[maxN / 2 + rowmid] *= ((1 - rowFactLarge*rowFactLarge) * rowFactSmall*rowFactSmall); //
		rowFactLarge = fixy(exp(-(maxN / 2)*(maxN / 2) *scaleLargeR));
		rowFactSmall = fixy(exp(-(maxN / 2)*(maxN / 2) *scaleSmallR));
		for (auto col = 1; col < maxN / 2; col++){
			auto backcol = maxN - col;
			auto colFactLarge = exp(-(col*col) * scaleLargeC);
			auto colFactSmall = exp(-(col*col) * scaleSmallC);
			filter[col] *= ((1 - colFactLarge) * colFactSmall);
			filter[backcol] *= ((1 - colFactLarge) * colFactSmall);
			filter[col + rowmid] *= ((1 - colFactLarge*rowFactLarge) * colFactSmall*rowFactSmall);
			filter[backcol + rowmid] *= ((1 - colFactLarge*rowFactLarge) * colFactSmall*rowFactSmall);
		}
		// loop along column 0 and expanded_width/2
		auto colFactLarge = fixy(exp(-(maxN / 2)*(maxN / 2) * scaleLargeC));
		auto colFactSmall = fixy(exp(-(maxN / 2)*(maxN / 2) * scaleSmallC));
		for (auto j = 1; j < maxN / 2; j++) {
			auto row = j * maxN;
			auto backrow = (maxN - j)*maxN;
			rowFactLarge = exp(-(j*j) * scaleLargeC);
			rowFactSmall = exp(-(j*j) * scaleSmallC);
			filter[row] *= ((1 - rowFactLarge) * rowFactSmall);
			filter[backrow] *= ((1 - rowFactLarge) * rowFactSmall);
			filter[row + maxN / 2] *= ((1 - rowFactLarge*colFactLarge) * rowFactSmall*colFactSmall);
			filter[backrow + maxN / 2] *= ((1 - rowFactLarge*colFactLarge) * rowFactSmall*colFactSmall);
		}
		filter[0] = (band.remove_dc) ? 0 : filter[0];
	}

	static int Power2RoundUp(int x)
	{
		if (x < 0)
		{
			return 0;
		}
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return x + 1;
	}

public:
	FourierFilter(const FourierFilter& that) = delete;
	FourierFilter() : ft_plan({ 0, false }), expanded_width(0){};
	~FourierFilter() { CufftPlanDealloc(ft_plan); }
	void Filter(thrust::device_vector<float>& input, const BandPassSettings& band, const FrameSize& frame)
	{
		if (band.do_band_pass == false)
		{
			return;
		}
		auto regen_ft = (old_size != frame);
		old_size = frame;
		if (regen_ft)
		{
			CufftPlanDealloc(ft_plan);
			auto maxE = std::max(frame.width, frame.height);
			expanded_width = Power2RoundUp(1.5*maxE);
			CufftSafeCall(cufftPlan2d(&ft_plan.first, expanded_width, expanded_width, CUFFT_C2C));
			//CufftSafeCall(cufftSetCompatibilityMode(cufft_plan_id, CUFFT_COMPATIBILITY_NATIVE));//lets hope nothing explodes 
			ft_plan.second = true;
		}
		if (regen_ft || (old_band != band))
		{
			auto filter_size = FrameSize(expanded_width, expanded_width);
			filter_h.assign(filter_size.n(), 1);
			GenerateBandFilter(filter_h, band, filter_size);
			//
			ThrustSafeResize(filter_d, filter_h.size());
			thrust::copy(filter_h.begin(), filter_h.end(), filter_d.begin());
			//should actually be abs_plus, but for a this filter they are equivalent 
			auto sum = thrust::reduce(filter_d.begin(), filter_d.end(), 0.0f, thrust::plus<float>());
			auto fix_up = 1.0f / sum;
			thrust::transform(filter_d.begin(), filter_d.end(), filter_d.begin(), ScaleByConstant(static_cast<float>(fix_up)));
			WriteDebugCuda(thrust::raw_pointer_cast(filter_d.data()), expanded_width, expanded_width, "filter_1_generated_filter.tif");
			old_band = band;
		}
		auto inplace_ptr = thrust::raw_pointer_cast(input.data());
		loadImage(img_expanded, inplace_ptr, frame, FrameSize(expanded_width, expanded_width));
		auto img_big_ptr = thrust::raw_pointer_cast(img_expanded.data());
		WriteDebugCuda(img_big_ptr, expanded_width, expanded_width, "filter_2_expanded.tif");
		//we're using C2C plans because some of the filtering (not shown here), breaks complex conjugate symetry
		CufftSafeCall(cufftExecC2C(ft_plan.first, img_big_ptr, img_big_ptr, CUFFT_FORWARD));//
		WriteDebugCuda(img_big_ptr, expanded_width, expanded_width, "filter_3_FT.tif");
		thrust::transform(img_expanded.begin(), img_expanded.end(), filter_d.begin(), img_expanded.begin(), ScaleMagnitude());
		WriteDebugCuda(img_big_ptr, expanded_width, expanded_width, "filter_4_FT_Filtered.tif");
		CufftSafeCall(cufftExecC2C(ft_plan.first, img_big_ptr, img_big_ptr, CUFFT_INVERSE));//
		WriteDebugCuda(thrust::raw_pointer_cast(img_expanded.data()), expanded_width, expanded_width, "filter_5_filtered_Image.tif");
		//
		auto real_output = true;
		GetBackImage(inplace_ptr, real_output, img_expanded, FrameSize(expanded_width, expanded_width), frame);//scale here
		WriteDebugCuda(inplace_ptr, frame.width, frame.height, "filter_6_cropped.tif");
	}
};

struct DPMSettings
{
	//specifies the DPM rectangle
	bool dpm_snap_bg;
	bool dpm_redo_bg;
	int dpm_left_column, dpm_top_row, dpm_width;
	bool isValid() const
	{
		return dpm_width > 0;
	}
	DPMSettings() : dpm_snap_bg(true), dpm_redo_bg(false), dpm_left_column(0), dpm_top_row(0), dpm_width(0){}
	DPMSettings(int U0, int V0, unsigned int P) :
		dpm_snap_bg(true), dpm_redo_bg(false), dpm_left_column(U0), dpm_top_row(V0), dpm_width(P)
	{

	}
	bool operator== (const DPMSettings &c1) const
	{
		return (dpm_left_column == c1.dpm_left_column && dpm_top_row == c1.dpm_top_row && dpm_width == c1.dpm_width);
	}
	bool operator!= (const DPMSettings &c1) const
	{
		return !(*this == c1);
	}
	friend std::ostream& operator<<(std::ostream& os, const DPMSettings& ia)
	{
		os << "(" << ia.dpm_left_column << "," << ia.dpm_top_row << ")";
		return os;
	}
	//For optimal performance the frame size should be a multiple of this, typically we crop the camera sensor
	//Typically GPU FFTs are limited by kernel switching time, so switching between the 2,3,4,5 radix kernels kills performance
	static const auto gpu_size_factor = 128;
	static int DPMSizeHint(int side)
	{
		return side / gpu_size_factor;
	}
	static FrameSize DPMSizeHint(const FrameSize& in)
	{
		auto h = DPMSizeHint(in.height);
		auto w = DPMSizeHint(in.width);
		return FrameSize(w, h);
	}
};

struct DPMBackground final : DPMSettings
{
	thrust::device_vector <cuComplex> img_small_d;
	thrust::device_vector <unsigned char > image_ft_log_d;
	std::vector<unsigned char> image_ft_log_h;// this gets wired to the GUI
	DPMBackground(){};
};

class DPM_GPU_Structs
{
	std::map<FrameSize, DPMBackground> backgrounds;
	thrust::device_vector<cuComplex> dpm_in_d;
	thrust::device_vector<cuComplex> dpm_in_ft_d;
	thrust::device_vector<cuComplex> dpm_small_img_temp_d;
	CufftPlanWrapper  C_C, C_K;//typdef as int, this way we do't include the cufft.h header 
	void DPM_Demux(thrust::device_vector<cuComplex>& out, const thrust::device_vector<unsigned short>& camera_frame, const FrameSize& info, DPMBackground& bg, bool compute_log);
	DPMSettings pos_old;
	DPMSettings pos_old_dpm_demux;//todo remove
	FrameSize input_size_old;
	FrameSize input_to_ft_old;
public:
	DPM_GPU_Structs(const DPM_GPU_Structs& that) = delete;
	DPM_GPU_Structs() : C_C({ 0, false }), C_K({ 0, false }){}
	~DPM_GPU_Structs()
	{
		CufftPlanDealloc(C_C);
		CufftPlanDealloc(C_K);
	}

	DPMBackground& getDPM_BG(const FrameSize& input_frame)
	{
		if (backgrounds.find(input_frame) == backgrounds.end())
		{
			backgrounds[input_frame] = DPMBackground();
		}
		return backgrounds.find(input_frame)->second;
	}
	void computeDPMPhase(thrust::device_vector<float>& out, const thrust::device_vector<unsigned short>& camera_frame, const FrameSize& info, DPMBackground& dpm);
};

//dirty hack
__device__ unsigned char val(unsigned char x)
{
	return x;
}

__device__ float val(cuComplex x)
{
	return log1p((x.x*x.x + x.y*x.y));
}

template<typename T>
__global__  void _FindCenterOfMass_H(float* sums, const T* in, int W, int spacing, int H)
{
	auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (x < W)
	{
		float sum = 0.0;
		for (auto y = 0; y < H; y++)
		{
			auto idx = y*spacing + x;
			sum += val(in[idx]);
		}
		sums[x] = sum;
	}
}

template<typename T>
__global__  void _FindCenterOfMass_W(float* sums, const T* in, int W, int spacing, int H)
{
	int y = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (y < H)
	{
		float sum = 0.0;
		for (auto x = 0; x < W; x++)
		{
			auto idx = y*spacing + x;
			sum += val(in[idx]);
		}
		sums[y] = sum;
	}
}

template<typename T>
DPMSettings FindCenterOfMass(const thrust::device_vector<T>& array, const DPMSettings& guess, int spacing, int H_original, int W_original)
{
	// Keep it Simple™
	// Todo replace with more thrust
	const auto blocksize = 64;
	dim3 threads(blocksize, 1);
	float *sumH_h, *sumC_h;
	auto H = guess.dpm_width;//its a square!
	auto W = guess.dpm_width;
	auto inrange = [](int x, int max){return (x > 0) && (x < max); };
	auto uo = guess.dpm_left_column;
	auto vo = guess.dpm_top_row;
	if (!inrange(vo, H_original) || !inrange(uo, W_original))
	{
		//maybe throw?
		return guess;
	}
	auto top_of_array_d = thrust::raw_pointer_cast(array.data());
	auto in = &top_of_array_d[vo*spacing + uo];
	{
		//Sum top to bottom for column max
		dim3 gridH(ceil(W / (1.0*threads.x)), 1);
		float* sumH_d;
		auto bytesH = W*sizeof(float);
		CudaSafeCall(cudaHostAlloc((void**)&sumH_h, bytesH, cudaHostAllocDefault));
		CudaSafeCall(cudaMalloc((void**)&sumH_d, bytesH));
		_FindCenterOfMass_H << <gridH, threads >> >(sumH_d, in, W, spacing, H);
		CudaSafeCall(cudaMemcpy(sumH_h, sumH_d, bytesH, cudaMemcpyDeviceToHost));
		CudaSafeCall(cudaFree(sumH_d));
	}
	{
		//Sum Left to Right for row max
		dim3 gridC(ceil(H / (1.0*threads.x)), 1);
		float* sumC_d;
		auto bytesC = H*sizeof(float);
		CudaSafeCall(cudaHostAlloc((void**)&sumC_h, bytesC, cudaHostAllocDefault));
		CudaSafeCall(cudaMalloc((void**)&sumC_d, bytesC));
		_FindCenterOfMass_W << <gridC, threads >> >(sumC_d, in, W, spacing, H);
		CudaSafeCall(cudaMemcpy(sumC_h, sumC_d, bytesC, cudaMemcpyDeviceToHost));
		CudaSafeCall(cudaFree(sumC_d));
	}
	//
	auto maxidx = [](const float* in, int N){return std::distance(in, std::max_element(in, in + N)); };
	uo = maxidx(sumH_h, W);
	vo = maxidx(sumC_h, H);
	//center
	uo = guess.dpm_left_column + uo;
	vo = guess.dpm_top_row + vo;
	//Now we want the corner
	auto fix = [](int value, int min, int max){return std::max(std::min(value, max), min); };
	uo = fix(uo - guess.dpm_width / 2, 0, W_original - guess.dpm_width / 2);
	vo = fix(vo - guess.dpm_width / 2, 0, H_original - guess.dpm_width / 2);
	CudaSafeCall(cudaFreeHost(sumH_h));
	CudaSafeCall(cudaFreeHost(sumC_h));
	auto mass = DPMSettings(uo, vo, guess.dpm_width);
	std::cout << "Center of Mass " << guess << "->" << mass << std::endl;
	return mass;
}

struct RangeScaling
{
	const float min_value_in, min_value_out;
	const float scaleR;
	RangeScaling(const float& Min_value_in, const float& Max_value_in, const float& Min_value_out, const float& Max_value_out) :
		min_value_in(Min_value_in), min_value_out(Min_value_out), scaleR((Max_value_out - min_value_out) / (Max_value_in - min_value_in))
	{
	}
	__host__ __device__
		float operator()(const cuComplex& a) const
	{
		auto h = min_value_out + (log1p(a.x*a.x + a.y*a.y) - min_value_in)*scaleR;
		return static_cast<unsigned char>(h);
	}
};

struct AngleFunctor
{
	__host__ __device__ float operator()(const cuComplex& a, const cuComplex& b) const
	{
		auto rx = (a.x*b.x + a.y*b.y);
		auto ry = (a.y*b.x - a.x*b.y);
		return atan2f(ry, rx);
	}
};

__global__ void _FillComplexAndShift(cuComplex* __restrict__ dst, const unsigned short* __restrict__ src,
	int spacin, int spaceout, int rows)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < spaceout) && (y < rows))
	{
		auto a = 1 - 2 * ((y + x) & 1);
		auto in = y*spacin + x;
		auto out = y*spaceout + x;
		float value = src[in];
		dst[out].x = value*a;
		dst[out].y = 0;
	}
}


void FillComplexAndShift(thrust::device_vector<cuComplex>& dst_vector, const thrust::device_vector<unsigned short>& src_vector, int spacein, int spaceout, int rows)
{
	if (spaceout > spacein)
	{
		throw std::runtime_error("Can't make data outa nowheres");
	}
	const auto blocksize = 32;//dies due to occupancy problems past 32, todo maybe replace with thrust
	dim3 threads(blocksize, blocksize);
	auto div = [](int W, int X){ return static_cast<int>(ceil(W / (1.0f*X))); };
	dim3 grid(div(spaceout, threads.x), div(rows, threads.y));
	auto dst = ThrustSafeGetPointer(dst_vector, spaceout*rows);
	auto src = thrust::raw_pointer_cast(src_vector.data());
	_FillComplexAndShift << <grid, threads >> >(dst, src, spacein, spaceout, rows);
}

void DPM_GPU_Structs::DPM_Demux(thrust::device_vector<cuComplex>& out, const thrust::device_vector<unsigned short>& camera_frame, const FrameSize& info, DPMBackground& dpm_bg, bool compute_log)
{
	auto C_C_changed = (info != input_to_ft_old);
	input_to_ft_old = info;
	//
	if (C_C_changed)
	{
		CufftPlanDealloc(C_C);
		CufftSafeCall(cufftPlan2d(&C_C.first, info.height, info.width, CUFFT_C2C));
		//CufftSafeCall(cufftSetCompatibilityMode(C_C.first, CUFFT_COMPATIBILITY_NATIVE));//Or else you need to pad the whole
		C_C.second = true;
	}
	//
	auto C_K_changed = (pos_old_dpm_demux != dpm_bg) || C_C_changed;
	pos_old_dpm_demux = static_cast<DPMSettings>(dpm_bg);
	if (C_K_changed)
	{
		std::cout << "Rebuilding Small DPM Buffer" << std::endl;
		CufftPlanDealloc(C_K);
		int n[2] = { dpm_bg.dpm_width, dpm_bg.dpm_width };//rows, cols
		int inembed[2] = { static_cast<int>(info.height), static_cast<int>(info.width) };//first element is unused
		int onembed[2] = { n[1], n[1] };
		auto istride = 1, ostride = 1;
		auto idist = n[0] * info.width;
		auto odist = n[1] * n[0];
		CufftSafeCall(cufftPlanMany(&C_K.first, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, 1));
		//CufftSafeCall(cufftSetCompatibilityMode(C_K.first, CUFFT_COMPATIBILITYdoDPMDemux_NATIVE));//Or else you need to pad the whole
		C_K.second = true;
	}
	//
	WriteDebugCuda(thrust::raw_pointer_cast(camera_frame.data()), info.width, info.height, "DPM_1_input.tif");
	//demux
	auto input_row_width = info.width;//some cameras add extra padding
	FillComplexAndShift(dpm_in_d, camera_frame, input_row_width, info.width, info.height);//int spacein, int spaceout, int cols, int rows
	WriteDebugCuda(thrust::raw_pointer_cast(dpm_in_d.data()), info.width, info.height, "DPM_2_complex_shifted.tif");
	auto img_ft = ThrustSafeGetPointer(dpm_in_ft_d, dpm_in_d.size());
	CufftSafeCall(cufftExecC2C(C_C.first, thrust::raw_pointer_cast(dpm_in_d.data()), img_ft, CUFFT_FORWARD));//can't be done inplace or else it will mess up zeropadding
	WriteDebugCuda(img_ft, info.width, info.height, "DPM_3_FT.tif");
	auto startft = &img_ft[dpm_bg.dpm_top_row*info.width + dpm_bg.dpm_left_column];
	auto out_ptr = ThrustSafeGetPointer(out, dpm_bg.dpm_width*dpm_bg.dpm_width);
	CufftSafeCall(cufftExecC2C(C_K.first, startft, out_ptr, CUFFT_INVERSE));
	WriteDebugCuda(out_ptr, dpm_bg.dpm_width, dpm_bg.dpm_width, "DPM_4_IFT.tif");
	//
	if (compute_log)
	{
		auto log_scale = RangeScaling(1, 20, 0, 255);
		ThrustSafeResize(dpm_bg.image_ft_log_d, dpm_in_ft_d.size());
		thrust::transform(dpm_in_ft_d.begin(), dpm_in_ft_d.end(), dpm_bg.image_ft_log_d.begin(), log_scale);
		dpm_bg.image_ft_log_h.resize(dpm_bg.image_ft_log_d.size());
		thrust::copy(dpm_bg.image_ft_log_d.begin(), dpm_bg.image_ft_log_d.end(), dpm_bg.image_ft_log_h.begin());
	}
}

void DPM_GPU_Structs::computeDPMPhase(thrust::device_vector<float>& out, const thrust::device_vector<unsigned short>& camera_frame, const FrameSize& info, DPMBackground& dpm_bg)
{
	if (dpm_bg.isValid() == false)
	{
		return;
	}
	auto pos_changed = (static_cast<DPMSettings>(dpm_bg) != pos_old);
	auto size_changed = ((info != input_size_old));
	auto redo_bg = (pos_changed || size_changed || dpm_bg.dpm_redo_bg);
	pos_old = static_cast<DPMSettings>(dpm_bg);
	input_size_old = static_cast<FrameSize>(info);
	if (redo_bg)
	{
		std::cout << "Rebuilding DPM Buffer" << std::endl;
		DPM_Demux(dpm_bg.img_small_d, camera_frame, info, dpm_bg, (dpm_bg.dpm_snap_bg == false));
		if (dpm_bg.dpm_snap_bg)
		{
			static_cast<DPMSettings&>(dpm_bg) = FindCenterOfMass(dpm_in_ft_d, dpm_bg, info.width, info.height, info.width);
			DPM_Demux(dpm_bg.img_small_d, camera_frame, info, dpm_bg, true);
			pos_old = static_cast<DPMSettings>(dpm_bg);
		}
	}
	DPM_Demux(dpm_small_img_temp_d, camera_frame, info, dpm_bg, false);
	ThrustSafeResize(out, dpm_small_img_temp_d.size());
	//todo, do this in-place
	thrust::transform(dpm_small_img_temp_d.begin(), dpm_small_img_temp_d.end(), dpm_bg.img_small_d.begin(), out.begin(), AngleFunctor());
	//
	WriteDebugCuda(thrust::raw_pointer_cast(dpm_bg.img_small_d.data()), dpm_bg.dpm_width, dpm_bg.dpm_width, "DPM_5_background.tif");
	WriteDebugCuda(thrust::raw_pointer_cast(dpm_small_img_temp_d.data()), dpm_bg.dpm_width, dpm_bg.dpm_width, "DPM_5_image.tif");
	WriteDebugCuda(thrust::raw_pointer_cast(out.data()), dpm_bg.dpm_width, dpm_bg.dpm_width, "DPM_5_angle.tif");
}

int main()
{
	//
	cudaFree(nullptr);//If it locks here, verify that arch matches your card
	auto image = ReadBuffer("dpm_1.tif");
	auto background = ReadBuffer("dpm_1_background.tif");
	DPMSettings dpm_settings_guess(693, 100, 384);
	BandPassSettings settings(0, 40, 0, 40, true, true);
	//
	DPM_GPU_Structs dpm_compute;
	FourierFilter fourier_filter;
	thrust::device_vector<float> phase_image_d;
	thrust::device_vector<unsigned short> fringe_image_d;
	//Simulate an acquisition
	auto runs = 100;
	size_t start = 0;
	for (auto i = 0; i < runs; i++)
	{
		auto&  input = (i == 0) ? background.img : image.img;
		ThrustSafeResize(fringe_image_d, input.size());
		thrust::copy(input.begin(), input.end(), fringe_image_d.begin());
		auto& dpm_bg = dpm_compute.getDPM_BG(image);
		if (i == 0)
		{
			static_cast<DPMSettings&>(dpm_bg) = dpm_settings_guess;
		}
		if (i == 1)
		{
			start = timestamp();
		}
		dpm_compute.computeDPMPhase(phase_image_d, fringe_image_d, image, dpm_bg);
		auto output_width = dpm_bg.dpm_width;
		fourier_filter.Filter(phase_image_d, settings, FrameSize(output_width, output_width));
		CudaSafeCall(cudaDeviceSynchronize());//this goes to the OGL, where the aspect ratio rescaling is handled
	}
	std::cout << "Cycle Time: " << (timestamp() - start) / static_cast<float> (runs - 1) << "ms" << std::endl;
	auto output_size = dpm_compute.getDPM_BG(image).dpm_width;
	WriteDebugCuda(thrust::raw_pointer_cast(phase_image_d.data()), output_size, output_size, "dpm_1_result.tif");
	//
	return 0;
}