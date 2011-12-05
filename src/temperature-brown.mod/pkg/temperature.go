package test

//#include "libmumax2.h"
import "C"

import (
	"mumax/gpu"
	"unsafe"
	"fmt"
)

func init(){
	fmt.Println("loaded test.mod")
}

// DEBUG: sets all values to their X (i) index
func SetIndexX(dst *gpu.Array) {
	partSize := dst.PartSize()
	C.setIndexX(
		(**C.float)(unsafe.Pointer(&(dst.Pointers()[0]))),
		C.int(partSize[0]),
		C.int(partSize[1]),
		C.int(partSize[2]))
}

// DEBUG: sets all values to their Y (j) index
func SetIndexY(dst *gpu.Array) {
	partSize := dst.PartSize()
	C.setIndexY(
		(**C.float)(unsafe.Pointer(&(dst.Pointers()[0]))),
		C.int(partSize[0]),
		C.int(partSize[1]),
		C.int(partSize[2]))
}

// DEBUG: sets all values to their Z (k) index
func SetIndexZ(dst *gpu.Array) {
	partSize := dst.PartSize()
	C.setIndexZ(
		(**C.float)(unsafe.Pointer(&(dst.Pointers()[0]))),
		C.int(partSize[0]),
		C.int(partSize[1]),
		C.int(partSize[2]))
}
