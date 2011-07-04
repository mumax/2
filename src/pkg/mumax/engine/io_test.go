//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


import (
	"testing"
	"mumax/host"
	"net"
	"reflect"
	"os"
	"exec"
)

func TestIO(test *testing.T) {
	size := []int{15, 19, 31}
	a1 := host.NewArray(2, size)
	for i := range a1.List {
		a1.List[i] = float32(i)
	}
	end1, end2 := net.Pipe()
	go Write(end1, a1)
	a2 := Read(end2)
	if !reflect.DeepEqual(a1, a2) {
		test.Fail()
	}
}


var t1, t2 *host.Array

func BenchmarkWriteHostArray(bench *testing.B) {
	size := []int{350, 190, 710}
	if t1 == nil {
		t1 = host.NewArray(3, size)
		bench.SetBytes(4 * int64((t1.Len())))
	}
	for i := 0; i < bench.N; i++ {
		f,_ := os.Open("iotest.t")
		Write(f, t1)
		f.Close()
	}
	exec.Command("rm", "-f", "iotest.t").Run()
}

//func BenchmarkRead(bench *testing.B) {
//	size := []int{3, 350, 190, 310}
//	if t2 == nil {
//		t2 = NewT4(size)
//		bench.SetBytes(4 * int64(Len(t2)))
//	}
//	for i := 0; i < bench.N; i++ {
//		t1.ReadFromF("iotest.t")
//	}
//}
