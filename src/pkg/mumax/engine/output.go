//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Functions to write tensors as binary data. 
// Intended for fast inter-process communication or data caching,
// not as a user-friendly format to store simulation output (use omf for that).
// Uses the machine's endianess.
//
// Author: Arne Vansteenkiste

import (
	//. "mumax/common"
	"mumax/host"
	"io"
	"unsafe"
)

const (
	T_MAGIC = 0x0A317423 // First 32-bit word of tensor blob. Identifies the format. Little-endian ASCII for "#t1\n"
)


// Utility function, reads from a named file instead of io.Reader.
//func WriteF(filename string, t host.Array) {
//	out := MustOpenWRONLY(filename)
//	defer out.Close()
//	bufout := bufio.NewWriter(out)
//	defer bufout.Flush()
//	Write(bufout, t)
//}


// Writes the array
func Write(out io.Writer, a *host.Array) {
	out.Write(IntToBytes(T_MAGIC))
	out.Write(IntToBytes(a.Rank()))
	for _, s := range a.Size {
		out.Write(IntToBytes(s))
	}
	for _, f := range a.List {
		out.Write((*[4]byte)(unsafe.Pointer(&f))[:]) // FloatToBytes() inlined for performance.
	}
}

// Converts the raw int data to a slice of 4 bytes
func IntToBytes(i int) []byte {
	return (*[4]byte)(unsafe.Pointer(&i))[:]
}

// Converts the raw float data to a slice of 4 bytes
func FloatToBytes(f float32) []byte {
	return (*[4]byte)(unsafe.Pointer(&f))[:]
}


// TODO: 
// also necessary to implement io.WriterTo, ReaderFrom
//func (t *T) WriteTo(out io.Writer) {
//	Write(out, t)
//}
