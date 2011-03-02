//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh

// This file implements reading and tokenizing input

import (
	"io"
	"container/vector"
)


// Reads a line and parses it into words
// empty lines are returned as empty string slices
func ReadLine(in io.Reader) (line []string, eof bool) {
	words_arr := [10]string{}
	words := vector.StringVector(words_arr[0:0])
	currword := ""
	for {
		char := ReadCharNoComment(in)

		if isEndline(char) {
			if currword != "" {
				words.Push(currword)
				currword = ""
			}
			eof = isEOF(char) && len(words) == 0
			line = []string(words)
			return
		}

		if isWhitespace(char) && currword != "" {
			words.Push(currword)
			currword = ""
		} // whitespace && currword == "": ignore whitespace

		if isCharacter(char) {
			currword += string(char)
		}
	}

	//not reached
	panic("Bug")
	return
}


// Reads one character from the Reader
// -1 means EOF
// errors are cought and cause panic
// TODO: flips on incomplete read from reader, should be BlockingReader
func ReadChar(in io.Reader) int {
	buffer := [1]byte{}
	switch nr, err := in.Read(buffer[0:]); true {
	case nr < 0: // error
		panic(err)
	case nr == 0: // eof TODO: or incomplete read!
		return -1
	case nr > 0: // ok
		return int(buffer[0])
	}
	panic(("ReadChar"))
	return 0 // never reached
}


// Reads a character from the Reader,
// ignoring bash-style comments (everything from a # till a line end)
func ReadCharNoComment(in io.Reader) int {
	char := ReadChar(in)
	if char == int('#') {
		for char != int('\n') && char != -1 {
			char = ReadChar(in)
		}
	}
	return char
}


// Reads and returns the first non-empty line
func ReadNonemptyLine(in io.Reader) (line []string, eof bool) {
	line, eof = ReadLine(in)
	for len(line) == 0 && !eof {
		line, eof = ReadLine(in)
	}
	return
}


func isEOF(char int) bool {
	return char == -1
}


func isEndline(char int) bool {
	if isEOF(char) || char == int('\n') || char == int(';') {
		return true
	}
	return false
}


func isWhitespace(char int) bool {
	if char == int(' ') || char == int('\t') || char == int(':') {
		return true
	}
	return false
}


func isCharacter(char int) bool {
	return !isEndline(char) && !isWhitespace(char)
}
