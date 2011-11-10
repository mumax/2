/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "tensor.h"
#include <iostream>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

using namespace std;

//_________________________________________________________________________________________________ new

/// @note this is the "master constructor". All other constructors should call this one
tensor* as_tensorN(float* list, int rank, int* size){
  assert(rank > -1);
  
  tensor* t = (tensor*)safe_malloc(sizeof(tensor));
  t->rank = rank;
  t->size = (int*)safe_calloc(rank, sizeof(int));   
  t->len = 1;
  
  for(int i=0; i<rank; i++){
    t->size[i] = size[i];
    t->len *= size[i];
  }
  t->list = list;
  t->flags = 0;
  assert(t->len > 0);
  return t;
}


tensor* new_tensor(int rank, ...){
  fprintf(stderr, "WARNING: new_tensor() is deprecated, use new_tensorN() instead\n");
  int* size = (int*)safe_calloc(rank, sizeof(int));
  
  va_list varargs;
  va_start(varargs, rank);
  
  int len = 1;
  for(int i=0; i<rank; i++){
   size[i] = va_arg(varargs, int);
   len *= size[i];
  }
  va_end(varargs);
  
  float* list = (float*)safe_calloc(len, sizeof(float));
 
  return as_tensorN(list, rank, size);
}


tensor* as_tensor(float* list, int rank, ...){
  
  int* size = (int*)safe_calloc(rank, sizeof(int));
  
  va_list varargs;
  va_start(varargs, rank);
  
  for(int i=0; i<rank; i++){
    size[i] = va_arg(varargs, int);
  }
  va_end(varargs);
  
  return as_tensorN(list, rank, size);
}

tensor* new_tensorN(int rank, int* size){
  
  int* tsize = (int*)safe_calloc(rank, sizeof(int));
  int len = 1;
  
  for(int i=0; i<rank; i++){
    tsize[i] = size[i];
    len *= size[i];
  }
 
  float* list = (float*)safe_calloc(len, sizeof(float));
  
  return as_tensorN(list, rank, tsize);
}

//_________________________________________________________________________________________________ util

void tensor_zero(tensor* t){
  for(int i = 0; i < t->len; i++){
    t->list[i] = 0.;
  }
}

void tensor_copy(tensor* source, tensor* dest){
  assert(tensor_equalsize(source, dest));
  for(int i=0; i<source->len; i++)
    dest[i] = source[i];
}

int tensor_equalsize(tensor* a, tensor* b){
  if(a->rank != b->rank){
    return 0;
  }
  for(int i=0; i < a->rank; i++){
    if(a->size[i] != b->size[i]){
      return 0;
    }
  }
  return 1;
}

int* tensor_size3D(int* size4D){
  assert(size4D[0] == 3);
  int* size3D = (int*)safe_calloc(3, sizeof(int));
  size3D[X] = size4D[1];
  size3D[Y] = size4D[2];
  size3D[Z] = size4D[3];
  return size3D;
}

int* tensor_size4D(int* size3D){
  int* size4D = (int*)safe_calloc(4, sizeof(int));
  size4D[0] = 3;
  size4D[1] = size3D[X];
  size4D[2] = size3D[Y];
  size4D[3] = size3D[Z];
  return size4D;
}

//_________________________________________________________________________________________________ access

float* tensor_get(tensor* t, int r ...){
  int* index = new int[t->rank];
  
  va_list varargs;
  va_start(varargs, r);
  if(r != t->rank){
    cerr << "2nd argument (" << r << ")!= tensor rank (" << t->rank << ")" << endl;
    exit(-3);
  }
  
  for(int i=0; i<t->rank; i++){
    index[i] = va_arg(varargs, int);
  }
  va_end(varargs);
  float* ret = tensor_elem(t, index);
  delete[] index;
  return ret;
}


int tensor_index(tensor* t, int* indexarray){
  int index = indexarray[0];
  assert(! (indexarray[0] < 0 || indexarray[0] >= t->size[0]));
  for (int i=1; i<t->rank; i++){
    assert(!(indexarray[i] < 0 || indexarray[i] >= t->size[i]));
    index *= t->size[i];
    index += indexarray[i];
  }
  return index;
}


float* tensor_elem(tensor* t, int* indexarray){
  return &(t->list[tensor_index(t, indexarray)]);
}


float** tensor_array2D(tensor* t){
  assert(t->rank == 2);
  return slice_array2D(t->list, t->size[0], t->size[1]);
}

float** slice_array2D(float* list, int size0, int size1){
  float** sliced = (float**)safe_calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = &list[i*size1];
  }
  return sliced;
}

float*** slice_array3D(float* list, int size0, int size1, int size2){
  float*** sliced = (float***)safe_calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = (float**)safe_calloc(size1, sizeof(float*));
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      sliced[i][j] = &list[ (i * size1 + j) *size2 + 0];
    }
  }
  return sliced;
}

float*** tensor_array3D(tensor* t){
  assert(t->rank == 3);
  return slice_array3D(t->list, t->size[0], t->size[1], t->size[2]);
}

float**** slice_array4D(float* list, int size0, int size1, int size2, int size3){
  float**** sliced = (float****)safe_calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = (float***)safe_calloc(size1, sizeof(float*));
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      sliced[i][j] = (float**)safe_calloc(size2, sizeof(float*));
    }
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      for(int k=0; k<size2; k++){
        sliced[i][j][k] = &list[ ((i * size1 + j) *size2 + k) * size3 + 0];
      }
    }
  }
  return sliced;
}

float**** tensor_array4D(tensor* t){
  assert(t->rank == 4);
  return slice_array4D(t->list, t->size[0], t->size[1], t->size[2], t->size[3]);
}


float***** slice_array5D(float* list, int size0, int size1, int size2, int size3, int size4){
  
  float***** sliced = (float*****)safe_calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = (float****)safe_calloc(size1, sizeof(float*));
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      sliced[i][j] = (float***)safe_calloc(size2, sizeof(float*));
    }
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      for(int k=0; k<size2; k++){
        sliced[i][j][k] = (float**)safe_calloc(size3, sizeof(float*));
      }
    }
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      for(int k=0; k<size2; k++){
        for(int l=0; l<size3; l++){
          sliced[i][j][k][l] = &list[ (((i * size1 + j) *size2 + k) * size3 + l) * size4 + 0];
        }
      }
    }
  }
  
  return sliced;
}

float***** tensor_array5D(tensor* t){
  assert(t->rank == 5);
  return slice_array5D(t->list, t->size[0], t->size[1], t->size[2], t->size[3], t->size[4]);
}

int tensor_length(tensor* t){
  int length = 1;
  for(int i=0; i < t->rank; i++){
    length *= t->size[i]; 
  }
  return length;
}


void delete_tensor(tensor* t){
  // for safety, we invalidate the tensor so we'd quickly notice accidental use after freeing.
  t->rank = -1;
  t->size = NULL;
  t->list = NULL;
  free(t->size);
  free(t->list);
  free(t);
}


void write_tensor(tensor* t, FILE* out){
  write_tensor_pieces(t->rank, t->size, t->list, out);
}


void write_tensor_fname(tensor* t, char* filename){
  FILE* file = fopen(filename, "wb");
  if(file == NULL){
    fprintf(stderr, "Could not write file: %s\n", filename);
    abort();
  }
  write_tensor(t, file);
  fclose(file);
}


/// First 32-bit word of tensor blob. Identifies the format. Little-endian ASCII for "#t1\n"
#define T_MAGIC 0x0A317423

void write_tensor_pieces(int rank, int* size, float* list, FILE* out){
  write_int(T_MAGIC, out);
  size_t length = 1;
  for(int i=0; i<rank; i++){
    length *= size[i];
  }  
  write_int(rank, out);
  for(int i=0; i<rank; i++){
    write_int(size[i], out);
  }
  size_t written = fwrite(list, sizeof(float), length, out);
  //error handling
  if(written != length){
    fprintf(stderr, "Could not write tensor data\n");
    abort();
  }
}


void write_int(int i, FILE* out){
  int32_t i32 = (int32_t)i;
  fwrite(&i32, sizeof(int32_t), 1, out);
}

int read_int(FILE* in){
  int32_t i;
  size_t n = fread(&i, sizeof(int32_t), 1, in);
  assert(n == 1);
  return (int)i;
}

void write_float(float f, FILE* out){
  fwrite(&f, sizeof(int32_t), 1, out);
}


void write_tensor_ascii(tensor* t, FILE* out){
  fprintf(out, "%d\n", t->rank);
  for(int i=0; i<t->rank; i++){
    fprintf(out, "%d\n", t->size[i]);
  }
  for(int i=0; i<tensor_length(t); i++){
    fprintf(out, "%e\n", t->list[i]);
  }
}


void format_tensor(tensor* t, FILE* out){
  // print rank...
  fprintf(out, "%d\n", t->rank);
  // ... and size...
  for(int i=0; i < t->rank; i++){
    fprintf(out, "%d\n", t->size[i]);
  }
  // ... and data
  for(int i=0; i < tensor_length(t); i++){
    fprintf(out, "% 11f ", t->list[i]);
    // If we reach the end of dimension, we print an extra newline
    // for each dimension:
    for(int j=0; j < t->rank; j++){
      // calc. the length in that dimension
      int size = 1;
      for(int k=j; k < t->rank; k++){
        size *= t->size[k];
      }
      // if we are at the end of the dimension, print the newline.
      if((i+1) % size == 0){
        fprintf(out, "\n");
      }
    }
  }
  
}


void print_tensor(tensor* t){
  write_tensor_ascii(t, stdout);
}

// TODO: catch file not found!
tensor* read_tensor(FILE* in){
  int rank = -1;
  int* size = NULL;
  float* list = NULL;
  read_tensor_pieces(&rank, &size, &list, in);
  return as_tensorN(list, rank, size);
}

tensor* read_tensor_fname(char* filename){
  FILE* file = fopen(filename, "rb");
  if(file == NULL){
    fprintf(stderr, "Could not read file: %s\n", filename);
    abort();
  }
  tensor* t = read_tensor(file);
  fclose(file);
  return t;
}


void read_tensor_pieces(int* rank, int** size, float** list, FILE* in){
  int magic = read_int(in);
  if(magic != T_MAGIC){
	fprintf(stderr, "Bad tensor header: %x\n", magic);
	abort();
  }
  (*rank) = read_int(in);
  (*size) = (int*)safe_calloc(*rank, sizeof(int));
  for(int i=0; i<(*rank); i++){
    (*size)[i] = read_int(in);
  }
  
  
  int length = 1;
  for(int i=0; i<*rank; i++){
    length *= (*size)[i];
  } 
  *list = (float*)safe_calloc(length, sizeof(float));
  size_t n = fread(*list, sizeof(float), length, in);
  assert(n == size_t(length));
}


tensor* tensor_component(tensor* t, int component){
  int* size = new int[t->rank-1];
  for(int i=0; i<t->rank-1; i++){
    size[i] = t->size[i+1];
  }
  
  //delete[] size;
  int* index = new int[t->rank];
  for(int i=1; i<t->rank; i++){
    index[i] = 0;
  }
  index[0] = component;
  
  tensor* slice = as_tensorN(tensor_elem(t, index), t->rank-1, size);
  delete[] index;
  return slice;
}


void delete_tensor_component(tensor* t){
  // for safety, we invalidate the tensor so we'd quickly notice accidental use after freeing.
  t->rank = -1;
  t->size = NULL;
  t->list = NULL;
  free(t->size);
  // we do not free t->list as it is owned by the parent tensor who may still be using it.
  free(t);
}


void* safe_malloc(int size){
  void* ptr = malloc(size);
  if(ptr == NULL){
    fprintf(stderr, "could not malloc(%d)\n", size);
    abort();
  }
  return ptr;
}

void* safe_calloc(int length, int elemsize){
  void* ptr = calloc(length, elemsize);
  if(ptr == NULL){
    fprintf(stderr, "could not calloc(%d, %d)\n", length, elemsize);
    abort();
  }
  return ptr;
}

#ifdef __cplusplus
}
#endif
