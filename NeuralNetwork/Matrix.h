#ifndef __NN_MATRIX_H__
#define __NN_MATRIX_H__

#include <cstdio>
#include <cctype>
#include <list>
#include <exception>

// #define NN_MATRIX_RUNTIME_CHECKS

namespace nn
{

class HumanReadableSize
{
public:
	HumanReadableSize(double size)
	{
		_size = size;
		buildString(_size);
	}
	
	HumanReadableSize(const char *str)
	{
		strcpy(_str, str);
		decodeString(_str);
	}
	
	HumanReadableSize(const std::string &str)
	{
		strcpy(_str, str.c_str());
		decodeString(_str);
	}
	
	double size() const
	{
		return _size;
	}
	
	const char *str() const
	{
		return _str;
	}
	
private:
	void buildString(double size)
	{
		static const char *_units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", NULL };
		
		int i = 0;
		while (size > 1024.0 && _units[i+1] != NULL)
		{
			size /= 1024.0;
			++i;
		}
		
		sprintf(_str, "%.2f %s", size, _units[i]);
	}

	void decodeString(const char *str)
	{
		static const char *_units = "BKMGTPEZY";
		
		char *endOfNumber;
		_size = strtod(str, &endOfNumber);
		
		while (*endOfNumber != '\0' && isspace(*endOfNumber))
			++endOfNumber;
		
		if (*endOfNumber != '\0')
		{
			const char *u = strchr(_units, *endOfNumber);
			
			if (u != NULL)
			{
				double f = pow(1024.0, (double)(u - _units));
				_size *= f;
			}
		}
	}
	
	double _size;
	char _str[64];
};

class MatrixMemoryAllocator
{
public:
	static MatrixMemoryAllocator *instance();
	
	void configure(uint32_t chunkSize);
	void reserve(uint32_t size);
	void releaseAll();
	
	uint8_t *allocate(uint32_t size);
	void release(uint8_t *v, uint32_t size);
	
	uint32_t getAllocatedSize() const;
	uint32_t getWaistedSize() const;
	
protected:
	static MatrixMemoryAllocator *_instance;
	
	uint32_t _chunkSize;
	
	struct Chunk
	{
		uint8_t *_begin;
		uint8_t *_end;
		uint8_t *_storageEnd;
		
		inline uint32_t totalSize() const { return _storageEnd - _begin; }
		inline uint32_t size() const { return _end - _begin; }
		inline uint32_t availableSize() const { return _storageEnd - _end; }
	};
	
	std::list<Chunk *> _fullChunks;
	std::list<Chunk *> _chunks;
};

template <class T> class MatrixT
{
public:
	using value_type = T;
	
	MatrixT()
	{
		_numRows = 0;
		_numColumns = 0;
		
		_m = nullptr;
	}
	
	MatrixT(int nrows, int ncolumn)
	{
		_numRows = nrows;
		_numColumns = ncolumn;
		
		// _m = new value_type[_numRows * _numColumns];
		_m = (value_type *)MatrixMemoryAllocator::instance()->allocate(sizeof(value_type) * _numRows * _numColumns);
		
		for (int i = 0; i < _numRows * _numColumns; ++i)
		{
			_m[i] = (value_type)0.0;
		}
	}
	
	MatrixT(const nn::MatrixT<T> &m) : MatrixT(m.numRows(), m.numColumns())
	{
		for (int i = 0; i < _numRows * _numColumns; ++i)
		{
			_m[i] = m._m[i];
		}
	}
	
	~MatrixT()
	{
		// delete [] _m;
		MatrixMemoryAllocator::instance()->release((uint8_t *)_m, sizeof(value_type) * _numRows * _numColumns);
	}
	
	inline int numRows() const { return _numRows; }
	inline int numColumns() const { return _numColumns; }
	
	void resize(int nrows, int ncolumn)
	{
		if (nrows != numRows() || ncolumn != numColumns())
		{
			// delete [] _m;
			MatrixMemoryAllocator::instance()->release((uint8_t *)_m, sizeof(value_type) * _numRows * _numColumns);
			
			_numRows = nrows;
			_numColumns = ncolumn;
			
			// _m = new value_type[_numRows * _numColumns];
			_m = (value_type *)MatrixMemoryAllocator::instance()->allocate(sizeof(value_type) * _numRows * _numColumns);
		}
		
		for (int i = 0; i < _numRows * _numColumns; ++i)
		{
			_m[i] = (value_type)0.0;
		}
	}
	
	inline value_type &operator () (int r, int c) { return _m[r * _numColumns + c]; }
	inline const value_type &operator () (int r, int c) const { return _m[r * _numColumns + c]; }
	
	inline value_type *ptr() { return _m; }
	inline const value_type *ptr() const { return _m; }
	
protected:
	int _numRows, _numColumns;
	value_type *_m;
};

using Matrix = MatrixT<float>;
using MatrixF = MatrixT<float>;
using MatrixD = MatrixT<double>;

template <class T> void add(const MatrixT<T> &a, const MatrixT<T> &b, MatrixT<T> &c)
{
#ifdef NN_MATRIX_RUNTIME_CHECKS
	if (a.numColumns() != b.numColumns() || a.numRows() != b.numRows())
		throw std::runtime_error("nn::add - a/b shape mismatch");
	
	if (a.numRows() != c.numRows() || a.numColumns() != c.numColumns())
		throw std::runtime_error("nn::add - c shape mismatch");
#endif
	
	for (int ir = 0; ir < c.numRows(); ++ir)
	{
		for (int ic = 0; ic < c.numColumns(); ++ic)
		{
			c(ir, ic) = a(ir, ic) + b(ir, ic);
		}
	}
}

template <class T> void subtract(const MatrixT<T> &a, const MatrixT<T> &b, MatrixT<T> &c)
{
#ifdef NN_MATRIX_RUNTIME_CHECKS
	if (a.numColumns() != b.numColumns() || a.numRows() != b.numRows())
		throw std::runtime_error("nn::subtract - a/b shape mismatch");
	
	if (a.numRows() != c.numRows() || a.numColumns() != c.numColumns())
		throw std::runtime_error("nn::subtract - c shape mismatch");
#endif
	
	for (int ir = 0; ir < c.numRows(); ++ir)
	{
		for (int ic = 0; ic < c.numColumns(); ++ic)
		{
			c(ir, ic) = a(ir, ic) - b(ir, ic);
		}
	}
}

template <class T> void copy(const MatrixT<T> &a, MatrixT<T> &b)
{
#ifdef NN_MATRIX_RUNTIME_CHECKS
	if (a.numColumns() != b.numColumns() || a.numRows() != b.numRows())
		throw std::runtime_error("nn::copy - a/b shape mismatch");
#endif
	
	for (int ir = 0; ir < b.numRows(); ++ir)
	{
		for (int ic = 0; ic < b.numColumns(); ++ic)
		{
			b(ir, ic) = a(ir, ic);
		}
	}
}

template <class T> void dot(const MatrixT<T> &a, const MatrixT<T> &b, MatrixT<T> &c)
{
#ifdef NN_MATRIX_RUNTIME_CHECKS
	if (a.numColumns() != b.numRows())
		throw std::runtime_error("nn::dot - a/b shape mismatch");
	
	if (a.numRows() != c.numRows() || b.numColumns() != c.numColumns())
		throw std::runtime_error("nn::dot - c shape mismatch");
#endif
	
	for (int ir = 0; ir < c.numRows(); ++ir)
	{
		for (int ic = 0; ic < c.numColumns(); ++ic)
		{
			T v = (T)0.0;
			for (int i = 0; i < a.numColumns(); ++i)
			{
				v += a(ir, i) * b(i, ic);
			}
			c(ir, ic) = v;
		}
	}
}

template <class T> void multiply(const MatrixT<T> &a, const MatrixT<T> &b, MatrixT<T> &c)
{
#ifdef NN_MATRIX_RUNTIME_CHECKS
	if (a.numColumns() != b.numColumns() || a.numRows() != b.numRows())
		throw std::runtime_error("nn::multiply - a/b shape mismatch");
	
	if (a.numRows() != c.numRows() || a.numColumns() != c.numColumns())
		throw std::runtime_error("nn::multiply - c shape mismatch");
#endif
	
	for (int ir = 0; ir < c.numRows(); ++ir)
	{
		for (int ic = 0; ic < c.numColumns(); ++ic)
		{
			T v = a(ir, ic) * b(ir, ic);
			c(ir, ic) = v;
		}
	}
}

template <class T> T sum(const MatrixT<T> &a, T s)
{
	for (int ir = 0; ir < a.numRows(); ++ir)
	{
		for (int ic = 0; ic < a.numColumns(); ++ic)
		{
			s += a(ir, ic);
		}
	}
	return s;
}

template <class T> T min(const MatrixT<T> &a, int &ir, int &ic)
{
	T v = a(0, 0);
	ir = 0;
	ic = 0;
	
	for (int _ir = 0; _ir < a.numRows(); ++_ir)
	{
		for (int _ic = 0; _ic < a.numColumns(); ++_ic)
		{
			if (a(_ir, _ic) < v)
			{
				v = a(_ir, _ic);
				ir = _ir;
				ic = _ic;
			}
		}
	}
	
	return v;
}

template <class T> T max(const MatrixT<T> &a, int &ir, int &ic)
{
	T v = a(0, 0);
	ir = 0;
	ic = 0;
	
	for (int _ir = 0; _ir < a.numRows(); ++_ir)
	{
		for (int _ic = 0; _ic < a.numColumns(); ++_ic)
		{
			if (a(_ir, _ic) > v)
			{
				v = a(_ir, _ic);
				ir = _ir;
				ic = _ic;
			}
		}
	}
	
	return v;
}

template <class T, class F> void map(MatrixT<T> &a, F f)
{
	for (int ir = 0; ir < a.numRows(); ++ir)
	{
		for (int ic = 0; ic < a.numColumns(); ++ic)
		{
			T &val = a(ir, ic);
			val = f(val);
		}
	}
}

template <class T, class F> void map(const MatrixT<T> &a, const MatrixT<T> &b, F f)
{
#ifdef NN_MATRIX_RUNTIME_CHECKS
	if (a.numColumns() != b.numColumns() || a.numRows() != b.numRows())
		throw std::runtime_error("nn::map - a/b shape mismatch");
#endif
	
	for (int ir = 0; ir < a.numRows(); ++ir)
	{
		for (int ic = 0; ic < a.numColumns(); ++ic)
		{
			T va = a(ir, ic);
			T vb = b(ir, ic);
			f(va, vb);
		}
	}
}

template <class T, class F> void imap(MatrixT<T> &a, F f)
{
	for (int ir = 0; ir < a.numRows(); ++ir)
	{
		for (int ic = 0; ic < a.numColumns(); ++ic)
		{
			T &val = a(ir, ic);
			val = f(ir, ic, val);
		}
	}
}

template <class T> void print(const MatrixT<T> &a, const char *label)
{
	printf("%s:\n", label);
	for (int ir = 0; ir < a.numRows(); ++ir)
	{
		printf("| ");
		for (int ic = 0; ic < a.numColumns(); ++ic)
		{
			if (ic != 0)
				printf("    ");
			printf("%.4f", a(ir, ic));
		}
		printf(" |\n");
	}
}

}; // namespace nn

#endif // __NN_MATRIX_H__
