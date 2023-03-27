// MIT Licensed
// Copyright @eudoxos
// Available at https://github.com/eudoxos/bytesize

#pragma once

#include<string>
#include<regex>
#include<iomanip>
#include<sstream>

namespace bytesize {
	class bytesize{
		// number of bytes
		size_t bytes;
	public:
		// construct from number
		bytesize(size_t bytes_): bytes(bytes_){}
		// parse from string
		static bytesize parse(const std::string& str){
			const static std::regex rx("\\s*(\\d+|\\d+[.]|\\d?[.]\\d+)\\s*((|ki|Mi|Gi|Ti|Pi|k|M|G|T|P)[Bb]?)\\s*");
			std::smatch m;
			if(!std::regex_match(str,m,rx)) throw std::runtime_error("Unable to parse '"+str+"' as size.");
			double d=std::stod(m[1].str());
			size_t mult=1;
			if(m[3]=="") mult=1;
			else if(m[3]=="ki") mult=1LL<<10;
			else if(m[3]=="Mi") mult=1LL<<20;
			else if(m[3]=="Gi") mult=1LL<<30;
			else if(m[3]=="Ti") mult=1LL<<40;
			else if(m[3]=="Pi") mult=1LL<<50;
			else if(m[3]=="k") mult=1'000LL;
			else if(m[3]=="M") mult=1'000'000LL;
			else if(m[3]=="G") mult=1'000'000'000LL;
			else if(m[3]=="T") mult=1'000'000'000'000LL;
			else if(m[3]=="P") mult=1'000'000'000'000'000LL;
			else throw std::logic_error("Unhandled prefix '"+m[2].str()+"'.");
			return bytesize(d*mult);
		}
		// represent as string
		std::string format() const {
			std::ostringstream oss;
			oss<<std::setprecision(3);
			if(bytes<1'000LL) oss<<bytes<<" B";
			else if(bytes<1'000'000LL) oss<<(bytes*1./1000LL)<<" kB";
			else if(bytes<1'000'000'000LL) oss<<(bytes*1./1000'000LL)<<" MB";
			else if(bytes<1'000'000'000'000LL) oss<<(bytes*1./1000'000'000LL)<<" GB";
			else if(bytes<1'000'000'000'000'000LL) oss<<(bytes*1./1000'000'000'000LL)<<" TB";
			else oss<<(bytes*1./1000'000'000'000'000LL)<<" PB";
			return oss.str();
		}
		// implicit conversion to size_t
		operator size_t(){ return bytes; }
		// implicit conversion to std::string
		operator std::string(){ return this->format(); }
	};
	// stream output operator
	inline std::ostream& operator<<(std::ostream& os, const bytesize& bs){ os<<bs.format(); return os; }
	// separate namespace for using namespace bytesize::literals;
	namespace literals{
		// bytes only with integer
		inline bytesize operator"" _B(unsigned long long int num){ return bytesize(num); }
		// floating-point numbers, like 5.5_kB
		inline bytesize operator"" _kiB(long double num){ return bytesize((size_t)((1LL<<10)*num)); }
		inline bytesize operator"" _MiB(long double num){ return bytesize((size_t)((1LL<<20)*num)); }
		inline bytesize operator"" _GiB(long double num){ return bytesize((size_t)((1LL<<30)*num)); }
		inline bytesize operator"" _TiB(long double num){ return bytesize((size_t)((1LL<<40)*num)); }
		inline bytesize operator"" _PiB(long double num){ return bytesize((size_t)((1LL<<50)*num)); }
		inline bytesize operator"" _kB(long double num){ return bytesize((size_t)(1'000LL*num)); }
		inline bytesize operator"" _MB(long double num){ return bytesize((size_t)(1'000'000LL*num)); }
		inline bytesize operator"" _GB(long double num){ return bytesize((size_t)(1'000'000'000LL*num)); }
		inline bytesize operator"" _TB(long double num){ return bytesize((size_t)(1'000'000'000'000LL*num)); }
		inline bytesize operator"" _PB(long double num){ return bytesize((size_t)(1'000'000'000'000'000LL*num)); }
		// repeated for integer literals so that e.g. 5_kB works
		inline bytesize operator"" _kiB(unsigned long long int num){ return bytesize((size_t)((1LL<<10)*num)); }
		inline bytesize operator"" _MiB(unsigned long long int num){ return bytesize((size_t)((1LL<<20)*num)); }
		inline bytesize operator"" _GiB(unsigned long long int num){ return bytesize((size_t)((1LL<<30)*num)); }
		inline bytesize operator"" _TiB(unsigned long long int num){ return bytesize((size_t)((1LL<<40)*num)); }
		inline bytesize operator"" _PiB(unsigned long long int num){ return bytesize((size_t)((1LL<<50)*num)); }
		inline bytesize operator"" _kB(unsigned long long int num){ return bytesize((size_t)(1'000LL*num)); }
		inline bytesize operator"" _MB(unsigned long long int num){ return bytesize((size_t)(1'000'000LL*num)); }
		inline bytesize operator"" _GB(unsigned long long int num){ return bytesize((size_t)(1'000'000'000LL*num)); }
		inline bytesize operator"" _TB(unsigned long long int num){ return bytesize((size_t)(1'000'000'000'000LL*num)); }
		inline bytesize operator"" _PB(unsigned long long int num){ return bytesize((size_t)(1'000'000'000'000'000LL*num)); }
	}
}
#define BYTESIZE_FMTLIB_FORMATTER \
	/* make bytesize::bytesize known to fmt::format */ \
	template<> struct fmt::formatter<bytesize::bytesize> { \
	  template<typename ParseContext> constexpr auto parse(ParseContext &ctx) { return ctx.begin(); } \
	  template<typename FormatContext> auto format(const bytesize::bytesize &bs, FormatContext &ctx) { return format_to(ctx.out(),"{}",bs.format()); } \
	};
