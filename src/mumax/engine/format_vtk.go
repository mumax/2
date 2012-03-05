//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// Add support for vtk 4.2 file output

package engine

// Author: Rémy Lassalle-Balier

import (
	"fmt"
	"io"
	. "mumax/common"
	"strings"
	"unsafe"
)

func init() {
	RegisterOutputFormat(&FormatVTK{})
}

// vtk output format
type FormatVTK struct{}

func (f *FormatVTK) Name() string {
	return "vtk"
}

func (f *FormatVTK) Write(out io.Writer, q *Quant, options []string) {
	dataformat := ""
	switch len(options) {
	case 0: // No options: default=ascii
		dataformat = "ascii"
	case 1:
		dataformat = strings.ToLower(options[0])
	default:
		panic(InputErr(fmt.Sprint("Illegal VTK options:", options)))
	}
	if dataformat == "text" {
		dataformat = "ascii"
	}
	dataformat = "ascii"

	writeVTKHeader(out, q)
	writeVTKPoints(out, dataformat)
	writeVTKCellData(out, q, dataformat)
	writeVTKFooter(out)
}

func writeVTKHeader(out io.Writer, q *Quant) {
	gridsize := GetEngine().GridSize()

	fmt.Fprintln(out, "<?xml version=\"1.0\"?>")
	fmt.Fprintln(out, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">")
	fmt.Fprintf(out, "\t<StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", gridsize[Z]-1, gridsize[Y]-1, gridsize[X]-1)
	fmt.Fprintf(out, "\t\t<Piece Extent=\"0 %d 0 %d 0 %d\">\n", gridsize[Z]-1, gridsize[Y]-1, gridsize[X]-1)
}

func writeVTKPoints(out io.Writer, dataformat string) {
	fmt.Fprintln(out, "\t\t\t<Points>")
	fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"points\" NumberOfComponents=\"3\" format=\"%s\">\n", dataformat)
	gridsize := GetEngine().GridSize()
	cellsize := GetEngine().CellSize()
	switch dataformat {
	case "ascii":
		for k := 0; k < gridsize[X]; k++ {
			for j := 0; j < gridsize[Y]; j++ {
				for i := 0; i < gridsize[Z]; i++ {
					x := (float32)(i) * (float32)(cellsize[Z])
					y := (float32)(j) * (float32)(cellsize[Y])
					z := (float32)(k) * (float32)(cellsize[X])
					_, err := fmt.Fprint(out, x, " ", y, " ", z, " ")
					if err != nil {
						panic(IOErr(err.String()))
					}
					//fmt.Fprint(out, " \n")
				}
			}
		}
	case "binary":
		// Conversion form float32 [4]byte in big-endian
		// encoding/binary is too slow
		// Inlined for performance, terabytes of data will pass here...
		var bytes []byte
		for k := 0; k < gridsize[X]; k++ {
			for j := 0; j < gridsize[Y]; j++ {
				for i := 0; i < gridsize[Z]; i++ {
					x := (float32)(i) * (float32)(cellsize[Z])
					y := (float32)(j) * (float32)(cellsize[Y])
					z := (float32)(k) * (float32)(cellsize[X])
					bytes = (*[4]byte)(unsafe.Pointer(&x))[:]
					out.Write(bytes)
					bytes = (*[4]byte)(unsafe.Pointer(&y))[:]
					out.Write(bytes)
					bytes = (*[4]byte)(unsafe.Pointer(&z))[:]
					out.Write(bytes)
				}
			}
		}
	default:
		panic(InputErr("Illegal VTK data format " + dataformat + ". Options are: ascii, binary"))
	}
	fmt.Fprintln(out, "</DataArray>")
	fmt.Fprintln(out, "\t\t\t</Points>")
}

func writeVTKCellData(out io.Writer, q *Quant, dataformat string) {
	N := q.NComp()
	data := q.Buffer().Array
	switch N {
	case SCALAR:
		fmt.Fprintf(out, "\t\t\t<PointData Scalars=\"%s\">\n", q.Name())
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n", q.Name(), SCALAR, dataformat)
	case VECTOR:
		fmt.Fprintf(out, "\t\t\t<PointData Vectors=\"%s\">\n", q.Name())
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n", q.Name(), VECTOR, dataformat)
	case SYMMTENS:
		fmt.Fprintf(out, "\t\t\t<PointData Tensors=\"%s\">\n", q.Name())
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n", q.Name(), TENS, dataformat)
	case TENS:
		fmt.Fprintf(out, "\t\t\t<PointData Tensors=\"%s\">\n", q.Name())
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n", q.Name(), TENS, dataformat)
	}
	gridsize := GetEngine().GridSize()
	switch dataformat {
	case "ascii":
		for i := 0; i < gridsize[X]; i++ {
			for j := 0; j < gridsize[Y]; j++ {
				for k := 0; k < gridsize[Z]; k++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == SYMMTENS {
						fmt.Fprint(out, data[SwapIndex(0, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(1, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(2, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(1, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(3, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(4, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(2, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(4, TENS)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(5, TENS)][i][j][k], " ")
					} else {
						for c := 0; c < N; c++ {
							fmt.Fprint(out, data[SwapIndex(c, N)][i][j][k], " ")
						}
						//fmt.Fprint(out, "\n")
					}
				}
			}
		}
	case "binary":
		// encoding/binary is too slow
		// Inlined for performance, terabytes of data will pass here...
		var bytes []byte
		for i := 0; i < gridsize[X]; i++ {
			for j := 0; j < gridsize[Y]; j++ {
				for k := 0; k < gridsize[Z]; k++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == SYMMTENS {
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(0, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(1, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(2, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(1, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(3, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(4, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(2, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(4, TENS)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(5, TENS)][i][j][k]))[:]
						out.Write(bytes)
					} else {
						for c := 0; c < N; c++ {
							bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(c, N)][i][j][k]))[:]
							out.Write(bytes)
						}
					}
				}
			}
		}
	default:
		panic(InputErr("Illegal VTK data format " + dataformat + ". Options are: ascii, binary"))
	}
	fmt.Fprintln(out, "</DataArray>")
	fmt.Fprintln(out, "\t\t\t</PointData>")
}

func writeVTKFooter(out io.Writer) {
	fmt.Fprintln(out, "\t\t</Piece>")
	fmt.Fprintln(out, "\t</StructuredGrid>")
	fmt.Fprintln(out, "</VTKFile>")
}
