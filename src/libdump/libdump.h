/*
 * libdump.h
 * 
 * Copyright 2012 Mykola Dvornik <mykola.dvornik@gmail.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 */

#ifndef _LIBDUMP_H_
#define _LIBDUMP_H_

#include <stdint.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

//~ // Header for dump data frame
//~ type Header struct {
	//~ Magic      string
	//~ Components int
	//~ MeshSize   [3]int
	//~ MeshStep   [3]float64
	//~ MeshUnit   string
	//~ Time       float64
	//~ TimeUnit   string
	//~ DataLabel  string
	//~ DataUnit   string
	//~ Precission uint64
//~ }

#define MEMBERWIDTH	 8
#define CRCLEN		 8

const char magic_word[] = "#dump002";

typedef struct {
	char 				magic[8];		// +0 
	uint64_t	 		components;		// +8
	uint64_t			mesh_size[3];	// +16
	double  			mesh_step[3];	// +40
	char				mesh_unit[8];	// +64
	double				time;			// +72
	char				time_unit[8];	// +80
	char				data_label[8];	// +88
	char 				data_unit[8];	// +96
	uint64_t			precision;		// +104
	void				*data;			// +112 
	// data is not aligned to 128! There will be performance penalty	
	// There is also 128 bit CRC at the end of data block
	
} dump_frame;

size_t HEADERLEN = sizeof(dump_frame);

typedef struct {
	uint16_t	number_of_frames;
	char		*file_name;
	dump_frame	*frame;
	uint64_t	*offsets;
	uint64_t 	*sizes;
	uint64_t	current_frame;
	uint64_t 	is_loaded;
} dump;


// Load the dump file. 
// Seek to the first frame. 
// Cache the offsets in case of multiple frames.
int ldInitDump(dump **dmp, const char* filename);

// Get the given frame
int ldGetFrame(dump **dmp, uint16_t i);

// Get the next frame
int ldGetNextFrame(dump **dmp);

// Get the previous frame
int ldGetPreviousFrame(dump **dmp);

#ifdef __cplusplus
}
#endif

#endif
