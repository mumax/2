package gpu

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

// DEBUG: sets all values to their X (i) index
func SetIndexX(dst *Array) {
	C.setIndexX(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		C.int(dst.size3D[0]),
		C.int(dst.partSize[1]),
		C.int(dst.size3D[2]))
}

// DEBUG: sets all values to their Y (j) index
func SetIndexY(dst *Array) {
	C.setIndexY(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		C.int(dst.size3D[0]),
		C.int(dst.partSize[1]),
		C.int(dst.size3D[2]))
}

// DEBUG: sets all values to their Z (k) index
func SetIndexZ(dst *Array) {
	C.setIndexZ(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		C.int(dst.size3D[0]),
		C.int(dst.partSize[1]),
		C.int(dst.size3D[2]))
}
