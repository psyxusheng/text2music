# /usr/local/bin/python3

import os,re
import librosa
import soundfile

def process_lrc(filename):
    try:
        content =  open(filename,'r',encoding='cp936').read()
    except:
        content =  open(filename,'r',encoding='utf8').read()
    lines = content.strip().split('\n')
    if len(lines) < 3:
        return None 
    ret = []
    for line in lines:
        try:
            start_time = re.findall('\[.+?\]',line)[0]
            words     = re.subn('\[.+?\]','',line)[0]
            ret.append([start_time[1:-1],words])
        except:
            continue
    results = []
    for i in range(len(ret)-1):
        start_time,words = ret[i]
        end_time,_ = ret[i+1]
        results.append([start_time , end_time  , words])
    return results

def timestring_to_secs(string):
    minute,string = string.split(':')
    sec,millsec = string.split('.')
    try:
        return float(minute)*60 + float(sec) + float(millsec)/1000
    except:
        return None



def find_song(root,fname):
    found = None
    for suffix in ['.mp3','.wav','.flac']:
        filename = os.path.join(root,fname+suffix)
        if os.path.exists(filename):
            return filename
    return found
            


def main(folder):
    for root,folder,fns in os.walk(folder):
        for fn in fns:
            fname,suffix = fn.split(".",-1)
            if suffix == 'lrc':
                try:
                    lines = process_lrc(os.path.join(root,fn))
                except:
                    continue
                if lines is not None:
                    # find the corresponding song
                    song_file = find_song(root , fname)
                    if song_file is not None:
                        audio,sr = librosa.load(song_file,sr = None , mono = True)

                        for st,ed,line in lines:

                            print('    ',st,ed,line)
                            
                            try:
                                st_time , ed_time = timestring_to_secs(st),timestring_to_secs(ed)
                                interval  = ed_time - st_time 
                            except:
                                continue
                            
                            if interval > 20 : continue

                            piece = audio[int(st_time * sr) : int(ed_time * sr)]
                            name = 'data/'+line+'.wav'
                            try:
                                soundfile.write(name,piece,sr)
                            except:
                                continue











if __name__ == '__main__':
    #out = main("/Users/mac/Downloads/周.杰.伦mp3+歌词文件/")
    #print(out)


    print(len(os.listdir('data')))