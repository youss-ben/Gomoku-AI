# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:17:06 2024

@author: youss
"""
import numpy as np
import math
import time

# Paramètres du jeu
BOARD_SIZE = 15
EMPTY = "."
BLACK = "X"  # Joueur 1
WHITE = "O"  # Joueur 2

# Plateau initial
def create_board():
    board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY)
    board[7][7] = BLACK  # Premier coup du joueur 1 au centre (H7)
    return board

# Affichage du plateau
def display_board(board, moves):
    print("  " + " ".join(map(str, range(BOARD_SIZE))))
    for i, row in enumerate(board):
        print(f"{chr(65 + i)} {' '.join(row)}")
    print("\nHistorique des coups :")
    for move in moves:
        print(f"{move[0]} : {chr(65 + move[1][0])}{move[1][1]}")

# Vérifie si un coup est valide
def is_valid_move(board, row, col, restricted_area=None):
    if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
        return False
    if board[row][col] != EMPTY:
        return False
    if restricted_area and (row, col) in restricted_area:
        return False
    return True

# Génération de la zone restreinte (7x7 autour de H7)
def generate_restricted_area():
    restricted_area = set()
    for i in range(4, 11):
        for j in range(4, 11):
            restricted_area.add((i, j))
    return restricted_area

# Vérifie si un joueur a gagné
def check_winner(board, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == player:
                for dr, dc in directions:
                    if all(
                        0 <= row + i * dr < BOARD_SIZE
                        and 0 <= col + i * dc < BOARD_SIZE
                        and board[row + i * dr][col + i * dc] == player
                        for i in range(5)
                    ):
                        return True
    return False

# Évaluation avancée des positions
def evaluate(board, player):
    opponent = BLACK if player == WHITE else WHITE
    score = 0

    # Alignements stratégiques
    score += count_alignments(board, player, 2) * 10
    score += count_alignments(board, player, 3) * 100
    score += count_alignments(board, player, 4) * 1000
    score -= count_alignments(board, opponent, 2) * 10
    score -= count_alignments(board, opponent, 3) * 100
    score -= count_alignments(board, opponent, 4) * 1000

    # Opportunités critiques
    score += detect_critical_patterns(board, player) * 500
    score -= detect_critical_patterns(board, opponent) * 500

    # Centralité
    score += centrality_score(board, player)

    return score

# Détection des patterns critiques (ex: alignements dangereux)
def detect_critical_patterns(board, player):
    patterns = [
        "11110",  # Alignement de 4 avec un espace libre
        "01111",  # Alignement de 4 avec un espace libre
        "10111", "11011",  # Menaces de 3 alignées
    ]
    score = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                for pattern in patterns:
                    if matches_pattern(board, row, col, player, pattern):
                        score += 50  # Ajouter un score pour chaque pattern
    return score

# Vérifie si une case correspond à un pattern donné
def matches_pattern(board, row, col, player, pattern):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        line = []
        for i in range(-4, 5):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                line.append(board[r][c])
            else:
                line.append(None)
        if "".join(str(1 if cell == player else 0 if cell == EMPTY else -1) for cell in line if cell is not None) == pattern:
            return True
    return False

# Compte les alignements consécutifs de longueur spécifiée
def count_alignments(board, player, length):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    count = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == player:
                for dr, dc in directions:
                    if all(
                        0 <= row + i * dr < BOARD_SIZE
                        and 0 <= col + i * dc < BOARD_SIZE
                        and board[row + i * dr][col + i * dc] == player
                        for i in range(length)
                    ):
                        count += 1
    return count

# Vérifie si un coup mène directement à une victoire
def is_winning_move(board, player, row, col):
    board[row][col] = player
    won = check_winner(board, player)
    board[row][col] = EMPTY
    return won

# Évaluation de la centralité
def centrality_score(board, player):
    center = BOARD_SIZE // 2
    score = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == player:
                score += max(0, 7 - abs(row - center)) * max(0, 7 - abs(col - center))
    return score

# Tri des coups pour exploration prioritaire
def sort_moves_by_interest(board, moves, player):
    move_scores = []
    for row, col in moves:
        board[row][col] = player
        score = evaluate(board, player)
        board[row][col] = EMPTY
        move_scores.append((score, (row, col)))
    move_scores.sort(reverse=True, key=lambda x: x[0])  # Tri décroissant
    return [move for _, move in move_scores]

# IA choisit le meilleur coup
def get_best_move(board, player, depth=3, time_limit=5):
    start_time = time.time()
    best_score = -math.inf
    best_move = None
    moves = generate_moves(board)
    
    # Vérification des victoires immédiates
    for row, col in moves:
        if is_winning_move(board, player, row, col):
            return row, col  # Priorité absolue aux coups gagnants

    # Vérification des blocages de victoires adverses
    opponent = BLACK if player == WHITE else WHITE
    for row, col in moves:
        if is_winning_move(board, opponent, row, col):
            return row, col  # Priorité aux blocages

    # Si aucune victoire immédiate, tri et exploration avec Minimax
    sorted_moves = sort_moves_by_interest(board, moves, player)
    for row, col in sorted_moves:
        board[row][col] = player
        score = minimax(board, depth - 1, -math.inf, math.inf, False, player, start_time, time_limit)
        board[row][col] = EMPTY
        if score > best_score:
            best_score = score
            best_move = (row, col)
        if time.time() - start_time > time_limit:
            break
    return best_move

# Implémentation Minimax avec élagage alpha-bêta
def minimax(board, depth, alpha, beta, maximizing_player, player, start_time, time_limit):
    if time.time() - start_time > time_limit or depth == 0:
        return evaluate(board, player)
    opponent = BLACK if player == WHITE else WHITE
    maximizing_score = -math.inf if maximizing_player else math.inf
    moves = generate_moves(board)
    sorted_moves = sort_moves_by_interest(board, moves, player if maximizing_player else opponent)
    for row, col in sorted_moves:
        board[row][col] = player if maximizing_player else opponent
        score = minimax(board, depth - 1, alpha, beta, not maximizing_player, player, start_time, time_limit)
        board[row][col] = EMPTY
        if maximizing_player:
            maximizing_score = max(maximizing_score, score)
            alpha = max(alpha, score)
        else:
            maximizing_score = min(maximizing_score, score)
            beta = min(beta, score)
        if beta <= alpha:
            break
    return maximizing_score

# Génère les coups valides autour des pions existants (réduction de l'espace de recherche)
def generate_moves(board):
    moves = set()
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY:
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == EMPTY:
                        moves.add((r, c))
    return list(moves)

# Demande une saisie valide pour la couleur du joueur
def get_valid_color_choice():
    while True:
        color = input("Choisissez votre couleur (X pour noir, O pour blanc): ").strip().upper()
        if color in {BLACK, WHITE}:
            return color
        print("Entrée invalide. Veuillez choisir 'X' pour noir ou 'O' pour blanc.")

# Demande une saisie valide pour un coup
def get_valid_move():
    while True:
        move = input("Entrez votre coup (ex: H7): ").strip().upper()
        if len(move) >= 2 and move[0].isalpha() and move[1:].isdigit():
            row = ord(move[0]) - 65
            col = int(move[1:])
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return row, col
        print("Entrée invalide. Veuillez entrer une lettre (A-O) suivie d'un chiffre (0-14) correspondant à une case valide.")

# Main game loop
def play_game():
    board = create_board()
    restricted_area = generate_restricted_area()
    human_player = get_valid_color_choice()
    ai_player = WHITE if human_player == BLACK else BLACK

    turn = BLACK  # Le joueur noir commence
    moves = [("Initial", (7, 7))]  # Historique des coups
    black_pieces, white_pieces = 1, 0  # Compteurs de pions

    # Gestion du tour initial
    if human_player == BLACK:
        print("Vous commencez. Premier coup forcé au centre (H7).")
        display_board(board, moves)
    else:
        print("L'IA commence avec le coup initial au centre (H7).")
        display_board(board, moves)

    print("Le joueur blanc joue maintenant.")
    if human_player == WHITE:
        while True:
            row, col = get_valid_move()
            if is_valid_move(board, row, col):
                board[row][col] = WHITE
                moves.append(("Humain", (row, col)))
                white_pieces += 1
                break
            print("Coup invalide. Essayez encore.")
    else:
        row, col = get_best_move(board, WHITE, depth=3, time_limit=5)
        print(f"L'IA joue: {chr(65 + row)}{col}")
        board[row][col] = WHITE
        moves.append(("IA", (row, col)))
        white_pieces += 1

    # Boucle principale du jeu
    turn = BLACK  # Retour au joueur noir
    while True:
        display_board(board, moves)

        if turn == human_player:
            print("C'est votre tour!")
            while True:
                row, col = get_valid_move()
                restricted = restricted_area if turn == BLACK and black_pieces == 1 else None
                if is_valid_move(board, row, col, restricted):
                    board[row][col] = human_player
                    moves.append(("Humain", (row, col)))
                    if turn == BLACK:
                        black_pieces += 1
                    else:
                        white_pieces += 1
                    break
                print("Coup invalide. Essayez encore.")
        else:
            print("L'IA réfléchit...")
            restricted = restricted_area if turn == BLACK and black_pieces == 1 else None
            row, col = get_best_move(board, ai_player, depth=3, time_limit=5)
            print(f"L'IA joue: {chr(65 + row)}{col}")
            board[row][col] = ai_player
            moves.append(("IA", (row, col)))
            if turn == BLACK:
                black_pieces += 1
            else:
                white_pieces += 1

        # Vérifie si quelqu'un a gagné
        if check_winner(board, turn):
            display_board(board, moves)
            print(f"{turn} a gagné!")
            break

        # Vérifie si les deux joueurs ont utilisé leurs 60 pions
        if black_pieces == 60 and white_pieces == 60:
            display_board(board, moves)
            print("Partie nulle : tous les pions ont été utilisés.")
            break

        # Passe au joueur suivant
        turn = WHITE if turn == BLACK else BLACK


if __name__ == "__main__":
    play_game()
