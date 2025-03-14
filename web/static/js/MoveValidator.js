class MoveValidator {
    constructor() {
        this.HEAD_POSITIONS = {
            white: 23,
            black: 11
        };
    }

    initialize(board, dice, currentPlayer, isFirstTurn = false, headMoveMade = false) {
        this.board = board;
        this.currentPlayer = currentPlayer;
        this.isFirstTurn = isFirstTurn;
        this.diceState = {
            original: [...dice],
            expanded: [...dice],
            remaining: [...dice],
            used: []
        };
        if (this.isDoublesRoll()) {
            this.diceState.expanded = Array(4).fill(dice[0]);
            this.diceState.remaining = Array(4).fill(dice[0]);
        }
        this.headMoveMade = headMoveMade;
        this.headMoveCount = headMoveMade ? 1 : 0;
        this.isFirstTurnSpecialDoubles = isFirstTurn && this.isDoublesRoll() && [3,4,6].includes(this.diceState.original[0]);
        this.maxHeadMoves = this.isFirstTurnSpecialDoubles ? 2 : 1;
    }

    updateDiceAfterMove(dieValue) {
        const dieIndex = this.diceState.remaining.indexOf(dieValue);
        if (dieIndex !== -1) {
            this.diceState.remaining.splice(dieIndex, 1);
            this.diceState.used.push(dieValue);
        }
    }

    updateAfterMove(fromPos, toPos) {
        const dieValue = this.getMoveDistance(fromPos, toPos);
        this.updateDiceAfterMove(dieValue);
        if (fromPos === this.HEAD_POSITIONS[this.currentPlayer]) {
            this.headMoveMade = true;
            this.headMoveCount = (this.headMoveCount || 0) + 1;
        }
        if (toPos === -1 || toPos === 'off') {
            this.board[fromPos] += (this.currentPlayer === 'white' ? -1 : 1);
        } else {
            this.board[fromPos] += (this.currentPlayer === 'white' ? -1 : 1);
            this.board[toPos] += (this.currentPlayer === 'white' ? 1 : -1);
        }
    }

    isDoublesRoll() {
        return (this.diceState.original.length === 2 && this.diceState.original[0] === this.diceState.original[1]);
    }

    ownsPieceAt(playerColor, pos) {
        return playerColor === 'white' ? this.board[pos] > 0 : this.board[pos] < 0;
    }

    canLandAt(playerColor, pos) {
        if (pos === 'off' || pos === -1) return true;
        return playerColor === 'white' ? this.board[pos] >= 0 : this.board[pos] <= 0;
    }

    getMoveDistance(fromPos, toPos) {
        if (toPos === 'off' || toPos === -1) return fromPos + 1;
        return fromPos - toPos;
    }

    canMoveFromHead() {
        if (this.isFirstTurnSpecialDoubles) {
            return (this.headMoveCount || 0) < this.maxHeadMoves;
        }
        return (this.headMoveCount || 0) === 0;
    }

    validateMove(fromPos, toPos) {
        if (toPos === 'off') toPos = -1;
        if (!this.ownsPieceAt(this.currentPlayer, fromPos)) {
            return { valid: false, error: `No ${this.currentPlayer} piece at position ${fromPos}` };
        }
        if (!this.canLandAt(this.currentPlayer, toPos)) {
            return { valid: false, error: "Cannot land on opponent's piece" };
        }
        if (fromPos === this.HEAD_POSITIONS[this.currentPlayer] && !this.canMoveFromHead()) {
            return { valid: false, error: "Only one checker may leave the head position per turn" };
        }
        const dieValue = this.getMoveDistance(fromPos, toPos);
        if (!this.diceState.remaining.includes(dieValue)) {
            return { valid: false, error: `No die with value ${dieValue} available` };
        }
        return { valid: true, error: null };
    }

    getValidMovesForPosition(fromPos) {
        if (!this.ownsPieceAt(this.currentPlayer, fromPos)) {
            return [];
        }
        if (fromPos === this.HEAD_POSITIONS[this.currentPlayer] && !this.canMoveFromHead()) {
            return [];
        }
        const validDestinations = [];
        for (const dieValue of this.diceState.remaining) {
            const toPos = fromPos - dieValue;
            if (toPos < 0) {
                if (toPos === -1 || fromPos + 1 === dieValue) {
                    validDestinations.push(-1);
                }
                continue;
            }
            if (this.canLandAt(this.currentPlayer, toPos)) {
                validDestinations.push(toPos);
            }
        }
        if (this.isDoublesRoll() && fromPos === 17) {
            if (this.diceState.remaining.includes(5) && this.canLandAt(this.currentPlayer, 12) && !validDestinations.includes(12)) {
                validDestinations.push(12);
            }
            if (this.diceState.remaining.includes(6) && this.canLandAt(this.currentPlayer, 11) && !validDestinations.includes(11)) {
                validDestinations.push(11);
            }
        }
        if (this.isDoublesRoll() && fromPos === 13) {
            if (this.diceState.remaining.includes(5) && this.canLandAt(this.currentPlayer, 8) && !validDestinations.includes(8)) {
                validDestinations.push(8);
            }
        }
        return validDestinations;
    }

    getAllValidMoves() {
        const validMovesByPiece = {};
        for (let pos = 0; pos < 24; pos++) {
            if (this.ownsPieceAt(this.currentPlayer, pos)) {
                const validMoves = this.getValidMovesForPosition(pos);
                if (validMoves.length > 0) {
                    validMovesByPiece[pos] = validMoves;
                }
            }
        }
        return validMovesByPiece;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = MoveValidator;
} else {
    window.MoveValidator = MoveValidator;
}
