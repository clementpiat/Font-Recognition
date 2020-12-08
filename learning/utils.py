import random as rd

def get_n_pairs(n, fonts):
    fonts_lenght = len(fonts)
    i = 0
    fonts_to_boundaries = {}
    while(i<fonts_lenght):
        font = fonts[i]
        j = i+1
        while(j<fonts_lenght and fonts[j] == font):
            j+=1

        fonts_to_boundaries[font] = (i,j-1)
        i = j

    indexes = []
    for i in range(n//2):
        index = rd.randint(0,fonts_lenght-1)
        font = fonts[index]
        
        index_neg = rd.randint(0,fonts_lenght-1)
        a,b = fonts_to_boundaries[font]
        index_pos = rd.randint(a,b)

        indexes += [(index,index_neg,int(fonts[index_neg]==font)), (index,index_pos,1)]

    # In case n is odd
    if(len(indexes)<n):
        index = rd.randint(0,fonts_lenght-1)
        index_neg = rd.randint(0,fonts_lenght-1)
        indexes.append((index,index_neg,int(fonts[index_neg]==fonts[index])))
    
    return indexes