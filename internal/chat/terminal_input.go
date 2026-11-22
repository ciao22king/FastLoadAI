package chat

import (
	"bufio"
	"io"
)

func readKeyEvents(reader io.Reader, out chan<- keyEvent) {
	buffered := bufio.NewReader(reader)
	var pending []rune
	next := func() (rune, error) {
		if len(pending) > 0 {
			r := pending[0]
			pending = pending[1:]
			return r, nil
		}
		r, _, err := buffered.ReadRune()
		return r, err
	}
	for {
		r, err := next()
		if err != nil {
			out <- keyEvent{kind: keyEOF}
			return
		}
		switch r {
		case '\r':
			out <- keyEvent{kind: keyEnter}
		case '\n':
			out <- keyEvent{kind: keyNewline}
		case 0x7f, 0x08:
			out <- keyEvent{kind: keyBackspace}
		case 0x03:
			out <- keyEvent{kind: keyCtrlC}
		case 0x04:
			out <- keyEvent{kind: keyCtrlD}
		case 0x01:
			out <- keyEvent{kind: keyCtrlA}
		case 0x05:
			out <- keyEvent{kind: keyCtrlE}
		case 0x0b:
			out <- keyEvent{kind: keyCtrlK}
		case 0x0c:
			out <- keyEvent{kind: keyCtrlL}
		case 0x15:
			out <- keyEvent{kind: keyCtrlU}
		case 0x17:
			out <- keyEvent{kind: keyCtrlW}
		case 0x09:
			out <- keyEvent{kind: keyTab}
		case 0x1b:
			event, leftover := readEscape(next)
			if leftover != 0 {
				pending = append([]rune{leftover}, pending...)
			}
			out <- event
		default:
			if r < 32 {
				out <- keyEvent{kind: keyIgnore}
			} else {
				out <- keyEvent{kind: keyRune, r: r}
			}
		}
	}
}

func readEscape(next func() (rune, error)) (keyEvent, rune) {
	r, err := next()
	if err != nil {
		return keyEvent{kind: keyEsc}, 0
	}
	if r != '[' && r != 'O' {
		return keyEvent{kind: keyEsc}, r
	}
	var sequence []rune
	for len(sequence) < 12 {
		r, err = next()
		if err != nil {
			break
		}
		sequence = append(sequence, r)
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || r == '~' {
			break
		}
	}
	return parseEscape(string(sequence)), 0
}

func parseEscape(sequence string) keyEvent {
	switch sequence {
	case "A":
		return keyEvent{kind: keyUp}
	case "B":
		return keyEvent{kind: keyDown}
	case "C":
		return keyEvent{kind: keyRight}
	case "D":
		return keyEvent{kind: keyLeft}
	case "H", "1~", "7~":
		return keyEvent{kind: keyHome}
	case "F", "4~", "8~":
		return keyEvent{kind: keyEnd}
	case "3~":
		return keyEvent{kind: keyDelete}
	case "5~":
		return keyEvent{kind: keyPageUp}
	case "6~":
		return keyEvent{kind: keyPageDown}
	case "Z":
		return keyEvent{kind: keyTab}
	default:
		return keyEvent{kind: keyIgnore}
	}
}

func wrapInput(input []rune, cursor, width int) ([][]rune, int, int) {
	if width < 1 {
		width = 1
	}
	var lines [][]rune
	current := []rune{}
	currentWidth := 0
	cursorRow, cursorRune := 0, 0
	found := false
	for index, r := range input {
		if r == '\n' {
			if index == cursor {
				cursorRow, cursorRune = len(lines), len(current)
				found = true
			}
			lines = append(lines, current)
			current = nil
			currentWidth = 0
			continue
		}
		runes := runeWidth(r)
		if currentWidth+runes > width && len(current) > 0 {
			lines = append(lines, current)
			current = nil
			currentWidth = 0
		}
		if index == cursor {
			cursorRow, cursorRune = len(lines), len(current)
			found = true
		}
		current = append(current, r)
		currentWidth += runes
	}
	if cursor == len(input) {
		cursorRow, cursorRune = len(lines), len(current)
		found = true
	}
	if !found {
		cursorRow, cursorRune = len(lines), len(current)
	}
	lines = append(lines, current)
	return lines, cursorRow, cursorRune
}
