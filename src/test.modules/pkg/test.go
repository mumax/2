
package test

//#include "../../libmumax2/libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

func TestFunc(dst, *Array) {
	C.test_func(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		C.int(dst.partLen4D))
}
