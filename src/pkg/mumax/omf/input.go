//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package omf

import (
	. "mumax/common"
	"mumax/tensor"
	"io"
	. "strings"
	"bufio"
	"fmt"
	"strconv"
	"unsafe"
)


func FRead(filename string) (info *Info, data *tensor.T4) {
	in := MustOpenRDONLY(filename)
	info, data = Read(in)
	return
}

func Read(in_ io.Reader) (info *Info, data *tensor.T4) {
	in := NewBlockingReader(bufio.NewReader(in_))
	info = ReadHeader(in)

	size := []int{3, info.Size[Z], info.Size[Y], info.Size[X]}
	data = tensor.NewT4(size)

	switch info.Format {
	default:
		panic("Unknown format: " + info.Format)
	case "text":
		readDataText(in, data)
	case "binary":
		switch info.DataFormat {
		default:
			panic("Unknown format: " + info.Format + " " + info.DataFormat)
		case "4":
			readDataBinary4(in, data)
		}
	}
	return
}


func Decode(in io.Reader) (t *tensor.T4, metadata map[string]interface{}) {
	info, data := Read(in)
	t = data
	metadata = info.Desc
	return
}


// omf.Info represents the header part of an omf file.
// TODO: add Err to return error status
// Perhaps CheckErr() func
type Info struct {
	Desc            map[string]interface{}
	Size            [3]int
	ValueMultiplier float32
	ValueUnit       string
	Format          string // binary or text
	DataFormat      string // 4 or 8
	StepSize        [3]float32
	MeshUnit        string
}


// Safe way to get Desc values: panics when key not present
func (i *Info) DescGet(key string) interface{} {
	value, ok := i.Desc[key]
	if !ok {
		panic("Key not found in Desc: " + key)
	}
	return value
}

// Safe way to get a float from Desc
func (i *Info) DescGetFloat32(key string) float32 {
	value := i.DescGet(key)
	fl, err := strconv.Atof32(value.(string))
	if err != nil {
		panic("Could not parse " + key + " to float32: " + err.String())
	}
	return fl
}


type File struct {
	Info
	*tensor.T4
}


func readDataText(in io.Reader, t *tensor.T4) {
	size := t.Size()[1:] // without the leading "3"
	data := t.Array()
	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < size[X]; i++ {
		for j := 0; j < size[Y]; j++ {
			for k := 0; k < size[Z]; k++ {
				for c := Z; c >= X; c-- {
					_, err := fmt.Fscan(in, &data[c][i][j][k])
					if err != nil {
						panic(err)
					}
				}
			}
		}
	}
}

func readDataBinary4(in io.Reader, t *tensor.T4) {

	size := t.Size()[1:] // without the leading "3"
	data := t.Array()

	var bytes4 [4]byte
	bytes := bytes4[:]

	in.Read(bytes)                                                                  // TODO: check for error, also on output (iotool.MustReader/MustWriter?)
	bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0] // swap endianess

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = 0.

	// Wicked conversion form float32 [4]byte in big-endian
	// encoding/binary is too slow
	// Inlined for performance, terabytes of data will pass here...
	controlnumber = *((*float32)(unsafe.Pointer(&bytes4)))
	// 	fmt.Println("Control number:", controlnumber)
	if controlnumber != CONTROL_NUMBER {
		panic("invalid control number: " + fmt.Sprint(controlnumber))
	}

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < size[X]; i++ {
		for j := 0; j < size[Y]; j++ {
			for k := 0; k < size[Z]; k++ {
				for c := Z; c >= X; c-- {
					n, err := in.Read(bytes) // TODO: check for error, also on output (iotool.MustReader/MustWriter?, have to block until all input is read)
					if err != nil || n != 4 {
						panic(err)
					}
					bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0] // swap endianess
					data[c][i][j][k] = *((*float32)(unsafe.Pointer(&bytes4)))
				}
			}
		}
	}

}


// INTERNAL: Splits "# key: value" into "key", "value"
func parseHeaderLine(str string) (key, value string) {
	strs := Split(str, ":", 2)
	key = Trim(strs[0], "# ")
	value = Trim(strs[1], "# ")
	return
}

// INTERNAL: true if line == "# begin_data"
func isHeaderEnd(str string) bool {
	str = ToLower(Trim(str, "# "))
	str = Replace(str, " ", "", -1)
	return HasPrefix(str, "begin:data")
}

// Parses the header part of the omf file
func ReadHeader(in io.Reader) *Info {
	desc := make(map[string]interface{})
	info := new(Info)
	info.Desc = desc

	line, eof := ReadLine(in)
	for !eof && !isHeaderEnd(line) {
		key, value := parseHeaderLine(line)

		switch ToLower(key) {
		default:
			panic("Unknown key: " + key)
			// ignored
		case "oommf", "segment count", "begin", "title", "meshtype", "xbase", "ybase", "zbase", "xstepsize", "ystepsize", "zstepsize", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "valuerangeminmag", "valuerangemaxmag", "end":
		case "xnodes":
			info.Size[X] = atoi(value)
		case "ynodes":
			info.Size[Y] = atoi(value)
		case "znodes":
			info.Size[Z] = atoi(value)
		case "valuemultiplier":
		case "valueunit":
		case "meshunit":
			// desc tags: parse further and add to metadata table
		case "desc":
			strs := Split(value, ":", 2)
			desc_key := Trim(strs[0], "# ")
			// Desc tag does not neccesarily have a key:value layout.
			// If not, we use an empty value string.
			desc_value := ""
			if len(strs) > 1 {
				desc_value = Trim(strs[1], "# ")
			}
			// 			fmt.Println(desc_key, " : ", desc_value)
			desc[desc_key] = desc_value
		}

		line, eof = ReadLine(in)
	}
	// the remaining line should now be the begin:data clause
	key, value := parseHeaderLine(line)
	value = TrimSpace(value)
	strs := Split(value, " ", 3)
	if ToLower(key) != "begin" || ToLower(strs[0]) != "data" {
		panic("Expected: Begin: Data")
	}
	info.Format = ToLower(strs[1])
	if len(strs) >= 3 { // dataformat for text is empty
		info.DataFormat = strs[2]
	}
	return info
}

func atoi(a string) int {
	i, err := strconv.Atoi(a)
	if err != nil {
		panic(err)
	}
	return i
}
