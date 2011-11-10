/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @file
 *
 * tensor provides a common interface for N-dimensional arrays of floats.
 * The tensor type knows its rank and sizes in each dimension,
 * so these do not have to be passed to function calls.
 * Tensors can also be input/output in a common format,
 * for sharing them between different programs or storing them in files.
 *
 * For an example how to use the tensor type, see tensor_test.cpp
 *
 * @todo ->array (void*)
 * @todo ->gpu?
 * @todo remove variadic functions, they sometimes give problems.
 *
 * @author Arne Vansteenkiste
 */
#ifndef TENSOR_H
#define TENSOR_H

// allow inclusion in C++ code
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#define X 0
#define Y 1
#define Z 2

//_____________________________________________________________________________________________________________ struct

typedef struct{
   int rank;         ///< the tensor's rank
   int* size;        ///< an array of length rank that stores the dimensions in all directions
   int len;          ///< total number of elements, equal to size[0] * size[1] * ... *size[rank-1]
   float* list;      ///< tensor data, stored as a (contiguous) array of length "len"
   void* array;      ///< the tensor data represented as a multi-dimensional array. underneath lies the data from "list".
   int flags;        ///< used to store some additional info like: allocated on RAM/GPU? symmetric? @deprecated ?
}tensor;

//_____________________________________________________________________________________________________________ new

/**
 * Creates a new tensor with given rank and size (as integer varargs).
 * Allocates the neccesary space for the elements.
 * Initializes with zeros.
 * @see new_tensorN()
 * @deprecated No compliation error arises when used with an int* as second argument.
 */
tensor* new_tensor(int rank,        ///< rank of the new tensor
                   ...              ///< sizes in each dimension, as many as rank
                   );

  
/**
 * The same as new_tensor(), but with the size given as an array.
 * This is only neccesary when the rank is not known at compile time,
 * otherwise just use new_tensor().
 * @see new_tensor()
 */
tensor* new_tensorN(int rank,     ///< rank of the new tensor
                    int* size     ///< sizes in each dimension, the number of elements should be equal to rank.
                    );



/**
 * Makes a tensor form existing data.
 * @see as_tensorN()
 */
tensor* as_tensor(float* list,   ///< array of data to wrapped in the tensor
                  int rank,      ///< rank of the new tensor
                  ...            ///< sizes in each dimension, as many as rank
                  );

                  
/**
 * Makes a tensor form existing data, without varargs.
 * @see as_tensor()
 */
tensor* as_tensorN(float* list,  ///< array of data to wrapped in the tensor
                   int rank,     ///< rank of the new tensor
                   int* size     ///< sizes in each dimension, as many as rank
                   );

//_____________________________________________________________________________________________________________ delete

/**
 * Frees the tensor, including its data list.
 * Make sure no pointers to that list exist anymore,
 * otherwise, delete_tensor_component() can be used to free everything but the data list.
 * @see delete_tensor_component()
 */
void delete_tensor(tensor* t     ///< the tensor to be deleted
);


/** 
 * Frees the tensor struct and all its members, except "list" (which points to the data array).
 * Tensors created by tensor_component() should not be freed with delete_tensor() 
 * but with delete_tensor_component(), as the parent tensor may still be using the data list. 
 * The same goes for tensors created with as_tensor() if a pointer to the data list still
 * exists, or for any tensor whose ->list has been stored somewhere.
 */
void delete_tensor_component(tensor* t);

//_____________________________________________________________________________________________________________ util

/**
 * Sets all the tensor's elements to zero.
 */
void tensor_zero(tensor* t);

/**
 * Copies source into dest. They should have equal sizes.
 */
void tensor_copy(tensor* source, tensor* dest);

/**
 * Checks whether both tensors have equal rank and size.
 * @return 1 for equal sizes, 0 otherwise.
 */
int tensor_equalsize(tensor* a, tensor* b);

/**
 * @internal Takes an array of 4 integers {3, N0, N1, N2} and returns the array {N0, N1, N2}
 */
int* tensor_size3D(int* size4D);

/**
 * @internal Takes an array of 3 integers {N0, N1, N2} and returns the array {3, N0, N1, N2}
 */
int* tensor_size4D(int* size3D);

//_____________________________________________________________________________________________________________ access

/** 
 * Makes a slice of a tensor, sharing the data with the original.
 * Example
 @code
  tensor* t = new_tensor(2, 3, 40); // 3 x 40 matrix
  tensor* t_row0 = tensor_component(t, 0); // points to row 0 of t
 @endcode
 */
tensor* tensor_component(tensor* t, int component);


/**
 * Returns the total number of elements in the tensor: size[0]*size[1]*...*size[rank-1].
 * This is also the length of the contiguous array that stores the tensor data.
 * @deprecated use tensor->len instead
 */
int tensor_length(tensor* t);


/**
 * Returns the address of element i,j,k,... inside the tensor.
 * This can be used to set or get elements form the tensor.
 * The index is checked against the tensor size and the code aborts when the index is out of bounds.
 * Of course, the "manual" way: t->list[i*size + j ...] can still be used as well.
 * @see tensor_get()
 */
float* tensor_elem(tensor* t, int* index);


/**
 * Same as tensor_elem(), but with varargs for ease of use.
 * The index is checked against the tensor size and the code aborts when the index is out of bounds.
 * Example:
 * @code
 *  tensor* t = new_tensor(2, 100, 100);
 *  *tensor_get(2, i, j) = 3.14;    
 *  float elemIJ = *tensor_get(2, i, j);
 * @endcode
 * @see tensor_elem()
 */
float* tensor_get(tensor* t, int rank, ...);


/**
 * Given an N-dimensional index (i, j, k, ...),
 * this function calculates the 1-dimensional
 * index in the corresponding array that stores the tensor data.
 * Thus, tensor_elem(i,j,k) is equivalent to list[tensor_index(i,j,k)].
 * The index is checked against the tensor size and the code aborts when the index is out of bounds.
 */
int tensor_index(tensor* t, int* indexarray);

//_____________________________________________________________________________________________________________ array

/**
 * An easy way to acces tensor's elements is to view it as an N-dimensional array.
 * Given a rank 2 tensor, this function returns a 2-dimensional array containing
 * the tensor's data. 
 */
float** tensor_array2D(tensor* t);

/**
 * @internal wraps the storage of a 1D array in a 2D array of given size.
 */
float** slice_array2D(float* list, int size0, int size1);

///@see tensor_array2D()
float*** tensor_array3D(tensor* t);

///@see slice_array2D()
float*** slice_array3D(float* list, int size0, int size1, int size2);

///@see tensor_array2D()
float**** tensor_array4D(tensor* t);

///@see slice_array2D()
float**** slice_array4D(float* list, int size0, int size1, int size2, int size3);

///@see tensor_array2D()
float***** tensor_array5D(tensor* t);

///@see slice_array2D()
float***** slice_array5D(float* list, int size0, int size1, int size2, int size3, int size4);

//_____________________________________________________________________________________________________________ I/O

/** 
 * Writes the tensor in binary format:
 * All data is written as 32-bit words, either integers or floats.
 * The first word is an integer that stores the rank N.
 * The next N words store the sizes in each of the N dimensions (also integers).
 * The remaining words are floats representing the data in row-major (C) order.
 *
 * @note currently the data is stored in the endianess of the machine. It might be nicer to store everything in big endian, though.
 */
void write_tensor(tensor* t, FILE* out);

/**
 * Writes to a file referenced by name instead of FILE*
 */
void write_tensor_fname(tensor* t, char* filename);

/**
 * Can be used as an alternative for write_tensor() if you don't want to use the tensor struct.
 */
void write_tensor_pieces(int rank, int* size, float* data, FILE* out);


/**
 * Reads the tensor from binary format.
 */
tensor* read_tensor(FILE* in);

/**
 * Reads from a file referenced by name instead of FILE*
 */
tensor* read_tensor_fname(char* filename);

/** Can be used as an alternative for read_tensor() if you don't want to use the tensor struct. */
void read_tensor_pieces(int* rank, int** size, float** list, FILE* in);

/** Prints the tensor as ascii text. 
 * The format is just the same as write_tensor(), but with ascii output instead of binary.
 * This can also be used to print to the screen: write_tensor_ascii(tensor, stdout);
 *
 * Todo: does anyone want a read_tensor_ascii() ?
 */
void write_tensor_ascii(tensor* t, FILE* out);

/** Prints the tensor in ascii text, but with nicer formatting: outputs a matrix in rows/columns, etc. */
void format_tensor(tensor* t, FILE* out);

/** Prints the tensor to standard output. */
void print_tensor(tensor* t);

void write_int(int i, FILE* out);
void write_float(float f, FILE* out);

//_____________________________________________________________________________________________________________ misc

/** A malloc that may print an error message instead of just causing a segfault when unable to allocate. */
void* safe_malloc(int size);

/** A malloc that may print an error message instead of just causing a segfault when unable to allocate. */
void* safe_calloc(int length, int elemsize);


#ifdef __cplusplus
}
#endif

#endif
