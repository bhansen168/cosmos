PATH = "WTH_2025.wtb"
#JOU file -- use "r",encoding="utf-8" -- list of ppl
#TRN file -- list of tournaments

import struct


def parse_wtb(file):
    gamesList=[]
    with open(file,"rb") as file1:
        all_bytes = file1.read()
    header = all_bytes[:16]
    gameData = all_bytes[16:]

    header_hex = [struct.unpack("<H", header[i:i+2])[0] for i in range(0,16,2)]
    #last 3 might be YYYY-MM-DD; otherwise irrelevant

    games = len(gameData)/68
    #print(f"GAMES: {games}")

    for i in range(0,len(gameData),68):
        data = gameData[i*68:(i+1)*68]

        if len(data[:6])>0:
            tournament, black, white = struct.unpack("<HHH", data[:6])

            black_score = data[6]
            theoretical = data[7] #how many perfect play would give black

            #print(tournament, black, white)
            #print(black_score, (64-black_score))

            moves = list(struct.unpack("<60B", data[8:68]))
            #convert moves to pairs
            pairs = [divmod(move,10) for move in moves] #column/10, row
            gamesList.append(pairs)
            
    return gamesList
        
        
        
    
    
    
if __name__ == "__main__":
    print("FIRST GAME: ",parse_wtb(PATH)[0])


    
