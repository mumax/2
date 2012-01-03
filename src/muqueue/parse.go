//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Command parsing

import (
"io"
)



//// Blocks until all requested bytes are read.
//type BlockingReader struct {
//	In io.Reader
//}
//
//func (r *BlockingReader) Read(p []byte) (n int, err os.Error) {
//	n, err = r.In.Read(p)
//	if err != nil {
//		if err == os.EOF {
//			return
//		} else {
//			panic(IOErr(err.String()))
//		}
//	}
//	if n < len(p) {
//		_, err = r.Read(p[n:])
//	}
//	if err != nil {
//		if err == os.EOF {
//			return
//		} else {
//			panic(IOErr(err.String()))
//		}
//	}
//	n = len(p)
//	return
//}
//
//func NewBlockingReader(in io.Reader) *BlockingReader {
//	return &BlockingReader{in}
//}

// Reads one character from the Reader.
// -1 means EOF.
// Errors are cought and cause panic
func readChar(in io.Reader) int {
	buffer := [1]byte{}
	switch nr, err := in.Read(buffer[0:]); true {
	case nr < 0: // error
		panic(err)
	case nr == 0: // eof
		return -1
	case nr > 0: // ok
		return int(buffer[0])
	}
	return 0 // never reached
}

//
func readLine(in io.Reader) (line string, eof bool) {
	char := readChar(in)
	eof = isEOF(char)

	for !isEndline(char) {
		line += string(byte(char))
		char = readChar(in)
	}
	return line, eof
}

func isEOF(char int) bool {
	return char == -1
}

func isEndline(char int) bool {
	return isEOF(char) || char == int('\n')
}
