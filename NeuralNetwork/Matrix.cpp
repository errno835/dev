#include "Matrix.h"

namespace nn
{

MatrixMemoryAllocator *MatrixMemoryAllocator::_instance = nullptr;

MatrixMemoryAllocator *MatrixMemoryAllocator::instance()
{
	if (_instance == nullptr)
	{
		_instance = new MatrixMemoryAllocator;
		_instance->_chunkSize = 16 * 1024 * 1024;
	}
	return _instance;
}

void MatrixMemoryAllocator::configure(uint32_t chunkSize)
{
	_chunkSize = chunkSize;
}

void MatrixMemoryAllocator::reserve(uint32_t size)
{
	_chunks.push_back(new Chunk());
	Chunk *c = _chunks.back();
	
	c->_begin = new uint8_t[size];
	c->_end = c->_begin;
	c->_storageEnd = c->_begin + size;
}

void MatrixMemoryAllocator::releaseAll()
{
	for (Chunk *c : _fullChunks)
	{
		delete [] c->_begin;
		delete c;
	}
	_fullChunks.clear();
	
	for (Chunk *c : _chunks)
	{
		delete [] c->_begin;
		delete c;
	}
	_chunks.clear();
}

uint8_t *MatrixMemoryAllocator::allocate(uint32_t size)
{
	if (size == 0)
		return nullptr;
	
	for (std::list<Chunk *>::iterator it = _chunks.begin(); it != _chunks.end(); ++it)
	{
		Chunk *c = *it;
		
		if (size <= c->availableSize())
		{
			uint8_t *v = c->_end;
			c->_end += size;
			
			if (c->availableSize() == 0)
			{
				_fullChunks.push_back(c);
				_chunks.erase(it);
			}
			
			return v;
		}
	}
	
	reserve(_chunkSize);
	uint8_t *v = allocate(size);
	return v;
}

void MatrixMemoryAllocator::release(uint8_t *v, uint32_t size)
{
}

}; // namespace nn
