#ifndef __NN_MATRIX_H__
#define __NN_MATRIX_H__

#include <cstdio>
#include <list>
#include <exception>

// #define NN_MATRIX_RUNTIME_CHECKS

namespace nn
{

class MatrixMemoryAllocator
{
public:
	static MatrixMemoryAllocator *instance();
	
	void configure(uint32_t chunkSize);
	void reserve(uint32_t size);
	void releaseAll();
	
	uint8_t *allocate(uint32_t size);
	void release(uint8_t *v, uint32_t size);
	
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
		// delete [] _m;
		MatrixMemoryAllocator::instance()->release((uint8_t *)_m, sizeof(value_type) * _numRows * _numColumns);
		
		_numRows = nrows;
		_numColumns = ncolumn;
		
		// _m = new value_type[_numRows * _numColumns];
		_m = (value_type *)MatrixMemoryAllocator::instance()->allocate(sizeof(value_type) * _numRows * _numColumns);
		
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

template <class T> void product(const MatrixT<T> &a, const MatrixT<T> &b, MatrixT<T> &c)
{
#ifdef NN_MATRIX_RUNTIME_CHECKS
	if (a.numColumns() != b.numRows())
		throw std::runtime_error("nn::product - a/b shape mismatch");
	
	if (a.numRows() != c.numRows() || b.numColumns() != c.numColumns())
		throw std::runtime_error("nn::product - c shape mismatch");
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
