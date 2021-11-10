import os, sys
import numpy as np

def main(audio_types, genders, kfold, path_origen, path_destino):
    
    genders_list = ["hombres", "mujeres"]
    types_required = []
    genders_required = []
    if audio_types == "all": types_required = ["phrase", "lhl", "l", "n", "h"]
    elif audio_types == "phrase": types_required = audio_types
    elif audio_types == "lhl": types_required = ["a_lhl", "i_lhl", "u_lhl"]
    elif audio_types == "aiu": types_required = ["a_l", "a_n", "a_h", "i_l", "i_n", "i_h", "u_l", "u_n", "u_h"]
    elif audio_types == "a": types_required = ["a_l", "a_n", "a_h"]
    elif audio_types == "i": types_required = ["i_l", "i_n", "i_h"]
    elif audio_types == "u": types_required = ["u_l", "u_n", "u_h"]
    elif audio_types == "aiu_n": types_required = ["a_n", "i_n", "u_n"]
    elif audio_types == "a_n": types_required = ["a_n"]
    elif audio_types == "i_n": types_required = ["i_n"]
    elif audio_types == "u_n": types_required = ["u_n"]
    else:
        print("audio type " + audio_types + " there is not this audio type")
        exit()

    if genders == "both": genders_required = genders_list
    else:
        if genders in genders_list: genders_required.append(genders)
        else:
            print("gender " + genders + " there is not this gender category")
            exit()

    path_origen = os.path.join(path_origen, "Saarbruecken")
    train = []
    test = []
    for k in range(kfold):
        train.append('{"labels": {"NORM": 0, "PATH": 1}, "meta_data": [')
        test.append('{"labels": {"NORM": 0, "PATH": 1}, "meta_data": [')
    
    labels_list = ["NORM", "PATH"]
    hombres, mujeres, norm, path = np.zeros((2,kfold)), np.zeros((2,kfold)), np.zeros((2,kfold)), np.zeros((2,kfold))
    k = 0
    for label in labels_list:
        for gender in genders_required:
            for audio in os.listdir(os.path.join(path_origen, label, gender)):
                #print(audio)
                temp = audio.split("-")
                speaker = temp[0]
                audio_type = temp[1].split(".")[0]
                
                if audio_type in types_required:
                    #print(audio)
                    # Add in one test fold and also in the rest train fold
                    path_audio = os.path.join(path_origen, label, gender, audio)
                    for i in range(0,kfold):
                        if i==k:
                            if label == 'NORM': norm[0,i]+=1
                            if label == 'PATH': path[0,i]+=1
                            if gender == 'hombres': hombres[0,i]+=1
                            if gender == 'mujeres': mujeres[0,i]+=1 
                            test[i] += '{"path": "' + path_audio + '", "label": "' + label + '", "speaker": "' + speaker + '"}, '
                            #print('test '+str(i))
                        else:
                            if label == 'NORM': norm[1,i]+=1
                            if label == 'PATH': path[1,i]+=1
                            if gender == 'hombres': hombres[1,i]+=1
                            if gender == 'mujeres': mujeres[1,i]+=1                            
                            train[i] += '{"path": "' + path_audio + '", "label": "' + label + '", "speaker": "' + speaker + '"}, '
                            #print('train '+str(i))
                    k+=1
                    if k==kfold: k=0 


    for k in range(0,kfold):
        # Save Test Json and Log
        test[k] = test[k][:-2]
        test[k] += ']}'
        f = open(path_destino+"/test_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write(test[k])
        f.close()
        f = open(path_destino+"/log/logtest_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write('Gender: male=%i, female=%i, total=%i\n' % (hombres[0,k], mujeres[0,k], hombres[0,k]+mujeres[0,k])) 
        f.write('Label: norm=%i, path=%i, total=%i' % (norm[0,k], path[0,k], norm[0,k]+path[0,k]))
        f.close()        

        # Save Train Json and Log
        train[k] = train[k][:-2]
        train[k] += ']}'
        f = open(path_destino+"/train_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write(train[k])
        f.close()
        f = open(path_destino+"/log/logtrain_"+audio_types+"_"+genders+"_"+"meta_data_fold"+str(k+1)+".json", "w")
        f.write('Gender: male=%i, female=%i, total=%i\n' % (hombres[1,k], mujeres[1,k], hombres[1,k]+mujeres[1,k])) 
        f.write('Label: norm=%i, path=%i, total=%i' % (norm[1,k], path[1,k], norm[1,k]+path[1,k]))
        f.close()

#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print('\nDescription: Create processing json list for s3prl baseline using Saarbruecken\n')
        print('Usage: Saarbruecken_preprocess.py audio_path label')
        print(' audio_types: Type of audio in Saarbruecken dataset: all | phrase | lhl | aiu | a | i | u | aiu_n | a_n | i_n | u_n ')
        print(' gender: gender of the speakers: hombres | mujeres | both')
        print(' kfold: k number of lists for crossvalidation (default: 1)')
        print(' audio_path (optional): Directory for audio files (default: data/audio)')
        print(' json_path (optional): Output directory (default: ./)\n')
        print('Example Dir: python Saarbruecken_preprocess.py phrase both\n')
    else:
        audio_types = args[0]
        genders = args[1]
        kfold = int(args[2])
        audio_path = 'data/audio'
        json_path = 'data/lst'
        if len(args)>3:
            audio_path = args[3]
        if len(args)>4:
            json_path = args[4]
        
        main(audio_types, genders, kfold, audio_path, json_path)

