package queue

import (
	"fmt"
)

type Node struct {
	host string
	gpus []*GPU
}

func NewNode(host string) *Node {
	n := new(Node)
	n.host = host
	n.gpus = make([]*GPU, 0)
	return n
}

func (n *Node) AddGPU(g *GPU) {
	n.gpus = append(n.gpus, g)
}

func (n *Node) String() string {
	return n.host + ":" + fmt.Sprint(n.gpus)
}
