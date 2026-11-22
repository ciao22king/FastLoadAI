package safetensors

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	ggufMagic   = "GGUF"
	ggufVersion = 3
	ggufAlign   = 32
)

const (
	typeUint32  = 4
	typeInt32   = 5
	typeFloat32 = 6
	typeBool    = 7
	typeString  = 8
	typeArray   = 9
)

const (
	ggmlF32  = 0
	ggmlF16  = 1
	ggmlF64  = 28
	ggmlBF16 = 30
	ggmlI32  = 26
	ggmlI64  = 27
)

type metadataValue struct {
	key   string
	kind  uint32
	value any
}

type outputTensor struct {
	name   string
	source *Tensor
	dims   []uint64
	kind   uint32
	size   int64
	offset uint64
}

func writeGGUF(path string, metadata []metadataValue, tensors []outputTensor) error {
	offset := int64(0)
	for index := range tensors {
		offset = align(offset, ggufAlign)
		tensors[index].offset = uint64(offset)
		offset += tensors[index].size
	}
	var header bytes.Buffer
	header.WriteString(ggufMagic)
	writeU32(&header, ggufVersion)
	writeU64(&header, uint64(len(tensors)))
	writeU64(&header, uint64(len(metadata)))
	for _, item := range metadata {
		writeString(&header, item.key)
		writeU32(&header, item.kind)
		if err := writeMetadataValue(&header, item); err != nil {
			return err
		}
	}
	for _, tensor := range tensors {
		writeString(&header, tensor.name)
		writeU32(&header, uint32(len(tensor.dims)))
		for _, dimension := range tensor.dims {
			writeU64(&header, dimension)
		}
		writeU32(&header, tensor.kind)
		writeU64(&header, tensor.offset)
	}
	if padding := align(int64(header.Len()), ggufAlign) - int64(header.Len()); padding > 0 {
		header.Write(make([]byte, padding))
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := bufio.NewWriterSize(file, 1<<20)
	if _, err := writer.Write(header.Bytes()); err != nil {
		return err
	}
	position := int64(0)
	for _, tensor := range tensors {
		target := int64(tensor.offset)
		if target > position {
			if _, err := writer.Write(make([]byte, target-position)); err != nil {
				return err
			}
			position = target
		}
		if err := writeTensorData(writer, tensor); err != nil {
			return fmt.Errorf("%s: %w", tensor.name, err)
		}
		position += tensor.size
	}
	return writer.Flush()
}

func writeTensorData(writer io.Writer, tensor outputTensor) error {
	source := tensor.source
	if tensor.kind != ggmlF32 || len(source.Shape) > 1 {
		_, err := io.CopyN(writer, io.NewSectionReader(source.Shard.file, source.DataStart, tensor.size), tensor.size)
		return err
	}
	sourceSize := source.DataEnd - source.DataStart
	raw := make([]byte, sourceSize)
	if _, err := source.Shard.file.ReadAt(raw, source.DataStart); err != nil {
		return err
	}
	if source.Dtype == "F32" {
		_, err := writer.Write(raw)
		return err
	}
	output := make([]byte, 0, tensor.size)
	var buffer [4]byte
	for index := 0; index < len(raw); {
		var bits uint32
		switch source.Dtype {
		case "BF16":
			if index+2 > len(raw) {
				return fmt.Errorf("truncated BF16 tensor")
			}
			bits = uint32(binary.LittleEndian.Uint16(raw[index:])) << 16
			index += 2
		case "F16":
			if index+2 > len(raw) {
				return fmt.Errorf("truncated F16 tensor")
			}
			bits = halfToFloat32(uint16(raw[index]) | uint16(raw[index+1])<<8)
			index += 2
		case "F64":
			if index+8 > len(raw) {
				return fmt.Errorf("truncated F64 tensor")
			}
			bits = math.Float32bits(float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[index:]))))
			index += 8
		default:
			return fmt.Errorf("cannot convert 1D tensor dtype %s", source.Dtype)
		}
		binary.LittleEndian.PutUint32(buffer[:], bits)
		output = append(output, buffer[:]...)
	}
	_, err := writer.Write(output)
	return err
}

func writeMetadataValue(buffer *bytes.Buffer, item metadataValue) error {
	switch item.kind {
	case typeUint32:
		writeU32(buffer, item.value.(uint32))
	case typeInt32:
		writeI32(buffer, item.value.(int32))
	case typeFloat32:
		writeF32(buffer, item.value.(float32))
	case typeBool:
		if item.value.(bool) {
			buffer.WriteByte(1)
		} else {
			buffer.WriteByte(0)
		}
	case typeString:
		writeString(buffer, item.value.(string))
	case typeArray:
		switch values := item.value.(type) {
		case []string:
			writeU32(buffer, typeString)
			writeU64(buffer, uint64(len(values)))
			for _, value := range values {
				writeString(buffer, value)
			}
		case []int32:
			writeU32(buffer, typeInt32)
			writeU64(buffer, uint64(len(values)))
			for _, value := range values {
				writeI32(buffer, value)
			}
		case []float32:
			writeU32(buffer, typeFloat32)
			writeU64(buffer, uint64(len(values)))
			for _, value := range values {
				writeF32(buffer, value)
			}
		default:
			return fmt.Errorf("unsupported array metadata type %T", item.value)
		}
	default:
		return fmt.Errorf("unsupported metadata type %d", item.kind)
	}
	return nil
}

func writeU32(writer io.Writer, value uint32) {
	var buffer [4]byte
	binary.LittleEndian.PutUint32(buffer[:], value)
	writer.Write(buffer[:])
}

func writeI32(writer io.Writer, value int32) {
	writeU32(writer, uint32(value))
}

func writeU64(writer io.Writer, value uint64) {
	var buffer [8]byte
	binary.LittleEndian.PutUint64(buffer[:], value)
	writer.Write(buffer[:])
}

func writeF32(writer io.Writer, value float32) {
	writeU32(writer, math.Float32bits(value))
}

func writeString(writer io.Writer, value string) {
	writeU64(writer, uint64(len(value)))
	io.WriteString(writer, value)
}

func align(value, alignment int64) int64 {
	return (value + alignment - 1) / alignment * alignment
}

func halfToFloat32(value uint16) uint32 {
	sign := uint32(value>>15) << 31
	exponent := uint32(value>>10) & 0x1f
	mantissa := uint32(value) & 0x3ff
	switch exponent {
	case 0:
		if mantissa == 0 {
			return sign
		}
		exponent = 1
		for mantissa&0x400 == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= 0x3ff
		exponent = exponent + (127 - 15)
	case 31:
		exponent = 255
	default:
		exponent = exponent + (127 - 15)
	}
	return sign | exponent<<23 | mantissa<<13
}
