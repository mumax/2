//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


import (
	"testing"
	. "mumax/common"
	"mumax/host"
	"reflect"
	"bufio"
	"os"
	"exec"
)

func TestIO(test *testing.T) {
	size := []int{15, 19, 31}
	a1 := host.NewArray(2, size)
	for i := range a1.List {
		a1.List[i] = float32(i)
	}
	f, err := os.Create("iotest.t")
	CheckErr(err, ERR_IO)
	Write(f, a1)
	f.Close()

	f, err = os.Open("iotest.t")
	CheckErr(err, ERR_IO)

	a2 := Read(f)
	f.Close()

	if !reflect.DeepEqual(a1, a2) {
		test.Fail()
	}
	exec.Command("rm", "-f", "iotest.t").Run()
}


var t1, t2 *host.Array

func BenchmarkWriteHostArray(bench *testing.B) {
	bench.StopTimer()
	size := []int{350, 190, 710}
	if t1 == nil {
		t1 = host.NewArray(3, size)
	}
	bench.SetBytes(4 * int64((t1.Len())))
	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		f, err := os.Create("iotest.t")
		CheckErr(err, ERR_BUG)
		b := bufio.NewWriter(f)
		Write(b, t1)
		b.Flush()
		f.Close()
	}
	bench.StopTimer()
	exec.Command("rm", "-f", "iotest.t").Run()
}

func BenchmarkReadHostArray(bench *testing.B) {
	bench.StopTimer()
	size := []int{350, 190, 710}
	if t1 == nil {
		t1 = host.NewArray(3, size)
	}
	bench.SetBytes(4 * int64((t1.Len())))
	f, err := os.Create("iotest.t")
	CheckErr(err, ERR_BUG)
	Write(f, t1)
	f.Close()
	bench.StartTimer()
	for i := 0; i < bench.N; i++ {
		f, _ := os.Open("iotest.t")
		t1 = Read(f)
		f.Close()
	}
	bench.StopTimer()
	exec.Command("rm", "-f", "iotest.t").Run()
}
