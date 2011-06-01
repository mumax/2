//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// File for playing around with the gob package.
// Author: Arne Vansteenkiste

import (
	"testing"
	"gob"
	"io"
	"bufio"
	"rand"
)


// benchmark transmission of a large float32 array through a gob encoder/decoder pair.
func BenchmarkGobTransmission(b *testing.B) {
	b.StopTimer()
	r, w := io.Pipe()
	dec := gob.NewDecoder(r)
	bufw := bufio.NewWriter(w)
	enc := gob.NewEncoder(bufw)

	N := 32 * 1024 * 1024
	in := make([]float32, N)

	// feeding gob random data instead of zeros really slows it down!
	for i := range in {
		in[i] = rand.Float32()
	}
	out := make([]float32, N)

	// warm-up gob
	go enc.Encode(in)
	dec.Decode(&out)

	b.SetBytes(int64(4 * N))
	go func() {
		for i := 0; i < b.N; i++ {
			enc.Encode(in)
			bufw.Flush()
		}
	}()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		dec.Decode(&out)
	}
}

// benchmark transmission of an array through a reader/writer pair for reference
func BenchmarkPipeTransmission(b *testing.B) {
	b.StopTimer()
	r, w := io.Pipe()

	N := 32 * 1024 * 1024 * 4
	in := make([]byte, N)
	out := make([]byte, N)

	b.SetBytes(int64(N))
	go func() {
		for i := 0; i < b.N; i++ {
			w.Write(in)
		}
	}()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		r.Read(out)
	}
}


// benchmark copy of a large float32 array for reference
func BenchmarkCopyTransmission(b *testing.B) {
	b.StopTimer()

	N := 32 * 1024 * 1024
	in := make([]float32, N)
	out := make([]float32, N)

	b.SetBytes(int64(4 * N))
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		copy(out, in)
	}
}

func TestGob(t *testing.T) {
	r, w := io.Pipe()
	dec := gob.NewDecoder(r)
	enc := gob.NewEncoder(w)

	N := 1024
	in := make([]float32, N)
	out := make([]float32, N)

	go enc.Encode(in)
	err2 := dec.Decode(&out)

	if err2 != nil {
		t.Fatal(err2)
	}
}
