/*
 * Copyright (c) 2018, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "MemoryManager.h"

#include <stdio.h>
#include <string.h>

namespace fpga {

MemoryManager::MemoryManager(void* _base, size_t _size) {
   base = _base;
   size = _size;
   printf("memory manager init, base: %p, size: %lu\n", base, size);
   memset(base, 0, size);
   chunks = new MemChunk((unsigned char*) base, size);
}

MemoryManager::~MemoryManager() {
   MemChunk* current = chunks;
   MemChunk* next = chunks;

   do {
      current = next;
      next = current->next;
      delete current;
   } while(next != nullptr);
}

void* MemoryManager::allocate(size_t allocSize) {
   std::lock_guard<std::mutex>   lock(this->memoryMutex);
   MemChunk* current = chunks;
   bool exp = true;
   size_t roundedSize = ((allocSize + 4095) / 4096) * 4096;
   do {
      if (current->free && roundedSize <= current->size) {
         //Try to acquire
         if (current->free.compare_exchange_weak(exp, false)) {
            //Split chunk if possible)
            if (current->size > roundedSize) {
               MemChunk* nchunk = new MemChunk(current->addr+roundedSize, current->size-roundedSize);
               current->size = roundedSize;
               nchunk->next = current->next;
               current->next = nchunk;
            }
            return current->addr;
         }
      }
      current = current->next;
   } while(current != nullptr);

   std::cerr << "Could not allocate chunk of size: " << roundedSize << std::endl;
   return nullptr;
}

void MemoryManager::free(void* ptr) {
   //TODO merge
   MemChunk* current = chunks;
   do {
      if (current->addr == ptr) {
         memset(current->addr, 0, current->size);
         current->free = true;
      }
      current = current->next;
   } while(current != nullptr);
}

} /* namespace fpga */
