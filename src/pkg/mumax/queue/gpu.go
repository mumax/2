package queue


import (
	"fmt"
)


type GPU struct {
	megabytes int
	busy      bool
}


func NewGPU(megabytes int) *GPU {
	return &GPU{megabytes, false}
}


func (g *GPU) String() string {
	return fmt.Sprint(g.megabytes, "MB,", busyStr(g.busy))
}

func busyStr(busy bool) string {
	if busy {
		return "busy"
	}
	return "free"
}
