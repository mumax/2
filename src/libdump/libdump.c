/*
 * libdump.c
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
 * 
 */
 
#include "libdump.h"

#ifdef __cplusplus
extern "C" {
#endif  

int	ldInitDump(dump **dmp, const char* filename){
	FILE *f;
	dump_frame buffer;
	
	uint64_t number_of_frames;
	
	uint64_t offset;
	uint64_t offsets_buffer[USHRT_MAX];
	uint64_t sizes_buffer[USHRT_MAX];
	
	uint64_t data_size;
	
	f = fopen(filename, "r");
	if (f == NULL) {
		fclose(f);
		return -1;
	}
	
	// Make sure we are in the beginning
	rewind(f);
	
	// Read at most HEADERLEN from the beginning of the file
	fread(&buffer, 1, HEADERLEN, f);
	
	if (strncmp(buffer.magic, magic_word, MEMBERWIDTH)) {
		fclose(f);
		return -1;
	}
	// cache the length of the first frame
	data_size = HEADERLEN + buffer.mesh_size[2] * buffer.mesh_size[1] * buffer.mesh_size[0] * buffer.components + CRCLEN;// Header + data + CRC
	if (buffer.precision == 4) {
		data_size *= sizeof(float);
	} 
	if (buffer.precision == 8) {
		data_size *= sizeof(double);
	}
	
	sizes_buffer[0] = data_size;
	offsets_buffer[0] = (uint64_t)0;
		
	number_of_frames = 1;
	while (!feof(f)) {
		// calculate the offset
		offset = buffer.mesh_size[2] * buffer.mesh_size[1] * buffer.mesh_size[0] * buffer.components + MEMBERWIDTH; // mesh_volume + CRC
		if (buffer.precision == 4) {
			offset *= sizeof(float);
		} 
		if (buffer.precision == 8) {
			offset *= sizeof(double);
		}
		fseek(f, offset, SEEK_CUR);
		offset = ftell(f);
		fread(&buffer, 1, HEADERLEN, f);
		if (strncmp(buffer.magic, magic_word, MEMBERWIDTH)) {
			// the file is broken, but we still have some useful frames
			break;
		}
		data_size = HEADERLEN + buffer.mesh_size[2] * buffer.mesh_size[1] * buffer.mesh_size[0] * buffer.components + CRCLEN;// Header + data + CRC
		if (buffer.precision == 4) {
			data_size *= sizeof(float);
		} 
		if (buffer.precision == 8) {
			data_size *= sizeof(double);
		}
		
		sizes_buffer[number_of_frames] = data_size;
		offsets_buffer[number_of_frames] = offset; // the offset is from the beginning of the file	
		number_of_frames++;
	}
	
	// seek to the beginning
	rewind(f);
	
	// get the first frame
	fread(&buffer, 1, data_size, f);
	fclose(f);
	
	(*dmp)->frame = &buffer;
	(*dmp)->is_loaded = 1;
	(*dmp)->number_of_frames = number_of_frames;
	
	(*dmp)->offsets = (uint64_t*)malloc(number_of_frames * sizeof(uint64_t));
	memcpy((*dmp)->offsets, offsets_buffer, number_of_frames * sizeof(uint64_t));
	
	(*dmp)->sizes = (uint64_t*)malloc(number_of_frames * sizeof(uint64_t));
	memcpy((*dmp)->sizes, sizes_buffer, number_of_frames * sizeof(uint64_t));
	
	(*dmp)->file_name = (char*)malloc(strlen(filename) * sizeof(char));
	memcpy((*dmp)->file_name, filename, strlen(filename) * sizeof(char));
	
	return 0;
}

int ldGetFrame(dump **dmp, uint16_t i) {
	FILE *f;	
	uint16_t current_frame;
	if ((*dmp)->frame == NULL || (*dmp)->file_name == NULL) {
		return 1;
	}
	current_frame = (*dmp)->current_frame;
	// get the current offset in the file
	f = fopen((*dmp)->file_name, "r");
	if (f == NULL) {
		fclose(f);
		return -1;
	}
	// Make sure we are in the beginning and seek to the frame
	rewind(f);
	fseek(f, (*dmp)->offsets[i], SEEK_SET);
	
	// Read the frame
	fread((*dmp)->frame, 1, (*dmp)->sizes[i], f);
	fclose(f);
	if (strncmp((*dmp)->frame->magic, magic_word, MEMBERWIDTH)) {	
		if ((*dmp)->frame != NULL) free((*dmp)->frame);
		ldGetFrame(dmp, current_frame);
		return -1;
	}
	(*dmp)->current_frame = i;
	return 0;
}

int ldGetNextFrame(dump **dmp) {
	uint16_t next_frame = (*dmp)->current_frame + 1;
	next_frame = (next_frame >= (*dmp)->number_of_frames) ? (*dmp)->number_of_frames - 1 : next_frame;
	return ldGetFrame(dmp, next_frame);
}

int ldGetPreviousFrame(dump **dmp) {
	uint16_t previous_frame = (*dmp)->current_frame - 1;
	previous_frame = (previous_frame == 0) ? 0 : previous_frame;
	return ldGetFrame(dmp, previous_frame);
}

#ifdef __cplusplus
}
#endif
