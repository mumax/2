package dump

// Magic number
const MAGIC = "#dump100"

// Precision identifier
const (
	FLOAT32 = 4
)

// Header+data frame.
type Frame struct {
	Header
	Data []float32
	CRC  uint64
}

// Header for dump data frame
type Header struct {
	TimeLabel  string
	Time       float64
	SpaceLabel string
	CellSize   [3]float64
	Rank       int
	Size       []int
	Precission uint64
}
