
__author__      = "Ankit Bisht"
__email__ = "bisht@knights.ucf.edu"


# importing all the required libraries

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import glob
import json
import re
from datetime import datetime
from datetime import timedelta


# opening and parsing all the json files 

dataset_list=[] # it contains all the data. All the data extracted for all the features for every event(JSON files) would be added to it
for filename in glob.glob('*.json'):
    filelist=[] # contains data of every feature for a single event(json file) at a time
    with open(filename, encoding="utf8") as json_data:
        
        data = json.load(json_data)
        
        list_publish_time=[] # list for time at which a tweet was published.
        List_unsorted_users=[]  # list for % of verified users in an interval
        List_unsorted_friends=[] # list for Average # of friends of users in an interval
        List_unsorted_followers=[] # list Average # of followers of users in an interval
        List_unsorted_posts=[] # List for Average # of posts of users in an interval
        list_tweet_length=[] # List contaning the length of tweet
        list_tweet_detail=[] # list contaning tweets  
        # all the above lists are unsorted
        
        i=0 # position of the data we are extracting
        for r in data['TweetsList']:
            
            #Appending data
            line= data["TweetsList"][i]["pubTime"]
            date_pattern = re.compile(r'[0-9]+:[0-9]+ [a-zA-Z]+ - [0-9]+ [a-zA-Z]+ [0-9]+',re.I)
            matches_date = date_pattern.findall(line)
            for match in matches_date:
                date_time= datetime.strptime(match, '%I:%M %p - %d %b %Y')
                list_publish_time.append(date_time)

            #Appending data
            user_tweets_1= data["TweetsList"][i]["userObj"]["verified"]        
            List_unsorted_users.append(user_tweets_1)
            
            #Appending data
            user_tweets_2= data["TweetsList"][i]["userObj"]["numFriends"]
            List_unsorted_friends.append(user_tweets_2)

            #Appending data
            user_tweets_3= data["TweetsList"][i]["userObj"]["numFollowers"]
            List_unsorted_followers.append(user_tweets_3)

            #Appending data
            user_tweets_4= data["TweetsList"][i]["userObj"]["numTweets"]
            List_unsorted_posts.append(user_tweets_4)

            #Appending data
            user_tweets_5= data["TweetsList"][i]["detail"]
            length_t = len(user_tweets_5.split())
            list_tweet_length.append(length_t)

            #Appending data
            detailofjson = user_tweets_5        
            list_tweet_detail.append(detailofjson)

            i+=1
    
    # sorting the lists
    sorted_List_time = list(list_publish_time)
    
    sorted_List_time.sort()
    sorted_list_tweet_length = [p for _,p in sorted(zip(list_publish_time,list_tweet_length))]
    sorted_list_tweet_detail = [p for _,p in sorted(zip(list_publish_time,list_tweet_detail))]
    List_sorted_users = [p for _,p in sorted(zip(list_publish_time,List_unsorted_users))]
    List_sorted_friends = [p for _,p in sorted(zip(list_publish_time,List_unsorted_friends))]
    List_sorted_followers = [p for _,p in sorted(zip(list_publish_time,List_unsorted_followers))]
    List_sorted_posts = [p for _,p in sorted(zip(list_publish_time,List_unsorted_posts))]
    
    # All the above lists are now sorted
    
    import math
    interval_duration = math.ceil(abs((sorted_List_time[-1] - sorted_List_time[0]).total_seconds()/3600.0)/50.0) # length of each interval for each event
   
    #**********************************IMPLEMENTING EACH FEATURE********************************
    
    #*****Average length of microblogs*****#                               
    
    ftilda_avg_microblogs_length=[]  # all the f_tilda...[] lists contain f ~(tilda) values which is equation (6) 
                                        #f~(t,k) = (f (t,k) − fk)/ σ(fk)

    def average_length_posts_of_user():

        list_avg_microblogs_length = [] # contains the average of length of tweets(posts) made in each interval or the Average length of microblogs

        begin_time=sorted_List_time[0]
        tweet_number=0
        tweet_info=0
        interval=1
        for i in range (1,51):  # because there are total 50 intervals

            end_time = begin_time+timedelta(hours=interval_duration)
            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                count=0
                sum_=0
                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        count +=1
                        sum_= sum_ + (sorted_list_tweet_length[tweet_info])
                        tweet_info+=1

                    begin_time += timedelta(hours=1)


                avg_microblog_length=round(sum_/count,2)
                #print ("Average length of microblogs in interval",interval,":",avg_microblog_length)
                list_avg_microblogs_length.append(avg_microblog_length)
                #print ("\n")
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_avg_microblogs_length.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1

        std_dev=np.std(list_avg_microblogs_length) # standard deviation
        mean_f=np.mean(list_avg_microblogs_length) # mean
        length_f = len(list_avg_microblogs_length)

        for i in range(length_f):

            f_tilda= (list_avg_microblogs_length[i]- mean_f)/std_dev
            ftilda_avg_microblogs_length.append(f_tilda)
    average_length_posts_of_user()       
    filelist.extend(ftilda_avg_microblogs_length)
    
    #****************number of positive (negative) words in microblogs**********#
    
    from collections import Counter # to count the number of words


    ftilda_positive_negative_words=[]
    def positive_negative_words():


        list_positive_negative_words=[]
        def readwords( filename ):
            f = open(filename)
            words = [ line.rstrip() for line in f.readlines()]
            return words

        positive = readwords('positive.txt') # this file contains most of the positive words of the English dictionary
        negative = readwords('negative.txt') # this file contains most of the negative words of the English dictionary

        begin_time= sorted_List_time[0]

        tweet_number= 0
        tweet_info= 0
        interval= 1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)
            total_words=0

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')

                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):


                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        count = Counter(tweet_detail.split())

                        pos = 0
                        neg = 0
                        for key, val in count.items():
                            key = key.rstrip('.,?!\n') # removing possible punctuation signs
                            if key in positive:
                                pos += val

                            if key in negative:
                                neg += val
                        total_words = total_words + neg + pos

                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                #print ("total_positive_negative_words",total_words)
                #print ("\n")
                list_positive_negative_words.append(total_words)
                begin_time = end_time
                interval +=1
            else:
                #print('Interval',interval, ': No tweets')
                list_positive_negative_words.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_positive_negative_words)
        mean_f=np.mean(list_positive_negative_words)
        length_f = len(list_positive_negative_words)
        for i in range(length_f):
            f_tilda= (list_positive_negative_words[i]- mean_f)/std_dev
            ftilda_positive_negative_words.append(f_tilda)
    positive_negative_words()
    filelist.extend(ftilda_positive_negative_words)
    
  #***************** % of microblogs with URL*****************###   
    
    ftilda_microblosg_url=[]
    def microblogs_with_url():
        list_microblogs_url = []


        begin_time=sorted_List_time[0]

        tweet_number= 0
        tweet_info= 0
        interval= 1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_wit_url = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        required_data = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet_detail)

                        if required_data:
                            microblogs_wit_url += 1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)

                percent = microblogs_wit_url/total_microblogs
                percentage_microblog_url= round(percent * 100,2)
                #print("percentage of microblogs with URL:",percentage_microblog_url,"%")
                list_microblogs_url.append(percentage_microblog_url)
                #print ("\n")
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_microblogs_url.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_microblogs_url)
        mean_f=np.mean(list_microblogs_url)
        length_f = len(list_microblogs_url)
        for i in range(length_f):
            f_tilda= (list_microblogs_url[i]- mean_f)/std_dev
            ftilda_microblosg_url.append(f_tilda)
    microblogs_with_url()
    filelist.extend(ftilda_microblosg_url)
    
 ##**************** % of microblogs with smiling (frowning) emoticons *********************##
            
    ftilda_microblogs_emoticons=[]
    def microblogs_with_emoticons():
        list_microblogs_emoticons=[]
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_wit_emoticons = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        required_data = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',tweet_detail)

                        if required_data:
                            #print(tweet_detail)
                            microblogs_wit_emoticons+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                percent = microblogs_wit_emoticons/total_microblogs
                percentage_of_emoticons = round(percent * 100,2)

                #print("percentage of microblogs with  smiling(frowning) emoticons:",percentage_of_emoticons,"%")
                list_microblogs_emoticons.append(percentage_of_emoticons)
                #print ("\n")
                begin_time = end_time
                interval += 1
            else:
                #print('Interval',interval, ': No tweets')
                list_microblogs_emoticons.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_microblogs_emoticons)
        mean_f=np.mean(list_microblogs_emoticons)
        length_f = len(list_microblogs_emoticons)
        for i in range(length_f):
            f_tilda= (list_microblogs_emoticons[i]- mean_f)/std_dev
            ftilda_microblogs_emoticons.append(f_tilda)
    microblogs_with_emoticons()
    filelist.extend(ftilda_microblogs_emoticons)
    
    
 #######************* % of positive (negative) microblogs**********************###########

    from textblob import TextBlob
    ftilda_positive_negaive_microblogs=[]
    def positive_negaive_microblogs():
        list_positive_negaive_microblogs = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)
            total_positive_negative = 0
            if (tweet_info <len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                negative_microblog_count = 0
                positive_microblog_count = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        required_data = TextBlob(tweet_detail)
                        if ((required_data.sentiment.polarity)>0):
                            #print (tweet_detail)
                            positive_microblog_count += 1
                        elif ((required_data.sentiment.polarity)<0):
                            negative_microblog_count += 1
                        tweet_info+=1

                    begin_time += timedelta(hours=1)

                positive_percent = positive_microblog_count/total_microblogs
                negative_percent = negative_microblog_count/total_microblogs
                total_positive_negative = positive_microblog_count + negative_microblog_count
                #print("total number of microblogs:",total_microblogs)
                #print("number of positive microblogs:",positive_microblog_count)
                #print("number of negative microblogs:",negative_microblog_count)
                #print ("total count:", total_positive_negative )
                #print("percentage of positive microblogs:",round(positive_percent * 100,2),"%")
                #print("percentage of negative microblogs:",round(negative_percent * 100,2),"%")
                list_positive_negaive_microblogs.append(total_positive_negative)
                #print ("\n")
                begin_time = end_time
                interval +=1
            else:
                #print('Interval',interval, ': No tweets')
                list_positive_negaive_microblogs.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_positive_negaive_microblogs)
        mean_f=np.mean(list_positive_negaive_microblogs)
        length_f = len(list_positive_negaive_microblogs)
        for i in range(length_f):
            f_tilda= (list_positive_negaive_microblogs[i]- mean_f)/std_dev
            ftilda_positive_negaive_microblogs.append(f_tilda)
    
    positive_negaive_microblogs()
    filelist.extend(ftilda_positive_negaive_microblogs)
    
 #####************% of microblogs with the first-person pronouns***********************#########
    ftilda_first_person_pronouns=[]
    def first_person_pronouns():
        list_first_person_pronouns = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                first_person_microblogs = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        # I don't think the below method is an optimal one because we may miss to include some pronouns 
                        list_ = ['I', 'i','me','me,','ME','we', 'my', 'mine', 'myself', 'us', 'our', 'ours', 'ourselves']
                        word_set = set(list_)
                        #key=tweet_detail.rstrip('.,?!\n') 
                        phrase_set = set(tweet_detail.split())
                        if word_set.intersection(phrase_set):
                            first_person_microblogs+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                percent = first_person_microblogs/total_microblogs
                #print("microblogs with first-person pronouns:",first_person_microblogs)
                #print("total number of microblogs:",total_microblogs)
                percentage = round(percent * 100,2)
                #print("percentage of microblogs with first-person pronouns :",percentage,"%")
                list_first_person_pronouns.append(percentage)
                #print ("\n")
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_first_person_pronouns.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_first_person_pronouns)
        mean_f=np.mean(list_first_person_pronouns)
        length_f = len(list_first_person_pronouns)
        for i in range(length_f):
            f_tilda= (list_first_person_pronouns[i]- mean_f)/std_dev
            ftilda_first_person_pronouns.append(f_tilda)
    
    first_person_pronouns()
    filelist.extend(ftilda_first_person_pronouns)
    
 ###************ % of microblogs with hashtags*********************########

    ftilda_microblogs_with_hashtags=[]
    def microblogs_with_hashtags():
        list_microblogs_with_hashtags = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_wit_hashtags = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        required_data = re.findall(r"#(\w+)", tweet_detail)

                        if required_data:
                            #print(tweet_detail)
                            microblogs_wit_hashtags+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                percent = microblogs_wit_hashtags/total_microblogs
                #print("microblogs with hashtags:",microblogs_wit_hashtags)
                #print("total number of microblogs:",total_microblogs)
                percentage=round(percent * 100,2)
                #print("percentage of microblogs with hashtags:",percentage,"%")
                list_microblogs_with_hashtags.append(percentage)
                #print ("\n")
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_microblogs_with_hashtags.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_microblogs_with_hashtags)
        mean_f=np.mean(list_microblogs_with_hashtags)
        length_f = len(list_microblogs_with_hashtags)
        for i in range(length_f):
            f_tilda= (list_microblogs_with_hashtags[i]- mean_f)/std_dev
            ftilda_microblogs_with_hashtags.append(f_tilda)
 
    microblogs_with_hashtags()
    filelist.extend(ftilda_microblogs_with_hashtags)
    
 ###************ % of microblogs with @ mentions *****************####
    ftilda_microblogs_with_at=[]
    def microblogs_with_at():
        list_microblogs_with_at=[]
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_wit_at = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        required_data = re.findall(r"@(\w+)", tweet_detail)

                        if required_data:
                            #print(tweet_detail)
                            microblogs_wit_at+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                percent = microblogs_wit_at/total_microblogs
                #print("microblogs with @:",microblogs_wit_at)
                #print("total number of microblogs:",total_microblogs)
                percentage=round(percent * 100,2)
                #print("percentage of microblogs with @:",percentage,"%")
                #print ("\n")
                list_microblogs_with_at.append(percentage)
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_microblogs_with_at.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1

        std_dev=np.std(list_microblogs_with_at)
        mean_f=np.mean(list_microblogs_with_at)
        length_f = len(list_microblogs_with_at)
        for i in range(length_f):
            f_tilda= (list_microblogs_with_at[i]- mean_f)/std_dev
            ftilda_microblogs_with_at.append(f_tilda)

    microblogs_with_at()
    filelist.extend(ftilda_microblogs_with_at)
    
 ####************ % of microblogs with question marks ****************######    
    ftilda_microblogs_with_question=[]
    def microblogs_with_question():
        list_microblogs_with_question = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_wit_question = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        required_data = '?' in tweet_detail[:-1]

                        if required_data:
                            #print(tweet_detail)
                            microblogs_wit_question+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                percent = microblogs_wit_question/total_microblogs
                #print("microblogs with question marks:",microblogs_wit_question)
                #print("total number of microblogs:",total_microblogs)
                percentage = round(percent * 100,2)
                #print("percentage of microblogs with question marks:", percentage ,"%")
                #print ("\n")
                list_microblogs_with_question.append(percentage)
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_microblogs_with_question.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_microblogs_with_question)
        mean_f=np.mean(list_microblogs_with_question)
        length_f = len(list_microblogs_with_question)
        for i in range(length_f):
            f_tilda= (list_microblogs_with_question[i]- mean_f)/std_dev
            ftilda_microblogs_with_question.append(f_tilda)
            
    microblogs_with_question()
    filelist.extend(ftilda_microblogs_with_question)
  ####********** % of microblogs with exclamation marks *************************###   
    ftilda_microblogs_with_exclamation=[]
    def microblogs_with_exclamation():
        list_microblogs_with_exclamation = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_wit_exclamation = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        #print(tweet_detail)
                        required_data = '!' in tweet_detail[:-1]

                        if required_data:
                            #print(tweet_detail)
                            microblogs_wit_exclamation+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                percent = microblogs_wit_exclamation/total_microblogs
                #print("microblogs with exclamation marks:",microblogs_wit_exclamation)
                #print("total number of microblogs:",total_microblogs)
                percentage = round(percent * 100,2)
                #print("percentage of microblogs with exclamation marks:",percentage,"%")
                #print ("\n")
                list_microblogs_with_exclamation.append(percentage)
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_microblogs_with_exclamation.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_microblogs_with_exclamation)
        mean_f=np.mean(list_microblogs_with_exclamation)
        length_f = len(list_microblogs_with_exclamation)
        for i in range(length_f):
            f_tilda= (list_microblogs_with_exclamation[i]- mean_f)/std_dev
            ftilda_microblogs_with_exclamation.append(f_tilda)
            
    microblogs_with_exclamation()
    filelist.extend(ftilda_microblogs_with_exclamation)
    
  #####*******  % of microblogs with multiple question/exclamation marks *****************###   
    
    ftilda_microblogs_with_question_exclamation=[]

    def microblogs_with_question_exclamation():
        list_microblogs_with_question_exclamation= []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_wit_question_exclamation = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= sorted_list_tweet_detail[tweet_info]
                        required_data_1 = tweet_detail.count('?')
                        required_data_2 = tweet_detail.count('!')

                        if required_data_1 > 1 or required_data_2 > 1:
                            #print(tweet_detail)
                            microblogs_wit_question_exclamation+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)

                #print ("Average length of microblogs in interval",t,":",sum/count)
                percent = microblogs_wit_question_exclamation/total_microblogs
                #print("microblogs with multiple question/exclamation marks:",microblogs_wit_question_exclamation)
                #print("total number of microblogs:",total_microblogs)
                percentage = round(percent * 100,2)
                #print("percentage of microblogs with questions/exclamation marks:",percentage ,"%")
                #print ("\n")
                list_microblogs_with_question_exclamation.append(percentage)
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_microblogs_with_question_exclamation.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_microblogs_with_question_exclamation)
        mean_f=np.mean(list_microblogs_with_question_exclamation)
        length_f = len(list_microblogs_with_question_exclamation)
        for i in range(length_f):
            f_tilda= (list_microblogs_with_question_exclamation[i]- mean_f)/std_dev
            ftilda_microblogs_with_question_exclamation.append(f_tilda)
   
    microblogs_with_question_exclamation()
    filelist.extend(ftilda_microblogs_with_question_exclamation)
    
  ######************ % of verified users *****************####   
    
    ftilda_verified_users=[]
    def verified_users():
        list_verified_users = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                microblogs_verified_users = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        total_microblogs +=1
                        tweet_detail= List_sorted_users[tweet_info]
                        required_data = tweet_detail

                        if required_data== True :
                            #print(tweet_detail)
                            microblogs_verified_users+=1
                        tweet_info+=1
                    begin_time += timedelta(hours=1)


                percent = microblogs_verified_users/total_microblogs
                #print("microblogs with verified users:",microblogs_verified_users)
                #print("total number of microblogs:",total_microblogs)
                percentage = round(percent * 100,2)
                #print("percentage of microblogs with verified users:",percentage ,"%")
                list_verified_users.append(percentage)
                #print ("\n")
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_verified_users.append(0)
                interval+=1
                end_time += timedelta(hours=1)
                begin_time = end_time
        std_dev=np.std(list_verified_users)
        mean_f=np.mean(list_verified_users)
        length_f = len(list_verified_users)
        for i in range(length_f):
            f_tilda= (list_verified_users[i]- mean_f)/std_dev
            ftilda_verified_users.append(f_tilda)

    verified_users()
    filelist.extend(ftilda_verified_users)
    
     ##************Average # of friends of users***********************###
    
    ftilda_users_friends = []
    def users_friends():
        list_users_friends = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                count=0
                sum_ =0
                microblogs_verified_users = 0
                total_microblogs=0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        count +=1
                        #print ('Tweet_',tweet_number,':','|||',List_sorted_friends[tweet_info])
                        sum_ = sum_+(List_sorted_friends[tweet_info])
                        tweet_info+=1
                    begin_time += timedelta(hours=1)

                total_count = round(sum_/count,2)
                #print ("Average # of friends of user",total_count)
                #print ("\n")
                list_users_friends.append(total_count)
                begin_time = end_time
                interval +=1
            else:
                #print('Interval',interval, ': No tweets')
                list_users_friends.append(0)
                interval+=1
                end_time += timedelta(hours=1)
                begin_time = end_time
        std_dev=np.std(list_users_friends)
        mean_f=np.mean(list_users_friends)
        length_f = len(list_users_friends)
        for i in range(length_f):
            f_tilda= (list_users_friends[i]- mean_f)/std_dev
            ftilda_users_friends.append(f_tilda)  

    users_friends()
    filelist.extend(ftilda_users_friends)
    
     ######****** Average # of followers of users ****************###
    
    ftilda_users_followers = []
    def users_followers():
        list_users_followers = []
        begin_time=sorted_List_time[0]

        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                count=0
                sum_ =0

                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        count +=1
                        #print ('Tweet_',tweet_number,':','|||',List_sorted_followers[tweet_info])
                        sum_ = sum_+(List_sorted_followers[tweet_info])
                        tweet_info+=1
                    begin_time += timedelta(hours=1)

                total_count = round(sum_/count,2)
                #print ("Average # of followers of user",total_count)
                list_users_followers.append(total_count)
                #print ("\n")
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_users_followers.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_users_followers)
        mean_f=np.mean(list_users_followers)
        length_f = len(list_users_followers)
        for i in range(length_f):
            f_tilda= (list_users_followers[i]- mean_f)/std_dev
            ftilda_users_followers.append(f_tilda)
         
    users_followers()
    filelist.extend(ftilda_users_followers)
    
    
     ###*********** Average # of posts of users    *****************####
    ftilda_users_posts = []
    def users_posts():

        list_users_posts=[]
        begin_time=sorted_List_time[0]
        tweet_number=0
        tweet_info=0
        interval =1
        for i in range (1,51):

            end_time = begin_time+timedelta(hours=interval_duration)

            if (tweet_info<len(sorted_List_time) and sorted_List_time[tweet_info]<=end_time):
                #print('Interval',interval, ':')
                count=0
                sum_ =0


                while (begin_time <= end_time and tweet_info<len(sorted_List_time)):
                    #print('before if z=',z)

                    if (sorted_List_time[tweet_info]<=end_time):
                        tweet_number +=1
                        count +=1
                        #print ('Tweet_',tweet_number,':','|||',List_sorted_posts[tweet_info])
                        sum_ = sum_+(List_sorted_posts[tweet_info])
                        tweet_info+=1
                    begin_time += timedelta(hours=1)

                avg_user_posts = round(sum_/count,2)
                #print ("Average # of posts of user",avg_user_posts)
                list_users_posts.append(avg_user_posts)
                #print ("\n")
                begin_time = end_time
                interval+=1
            else:
                #print('Interval',interval, ': No tweets')
                list_users_posts.append(0)
                end_time += timedelta(hours=1)
                begin_time = end_time
                interval+=1
        std_dev=np.std(list_users_posts)
        mean_f=np.mean(list_users_posts)
        length_f = len(list_users_posts)
        for i in range(length_f):
            f_tilda= (list_users_posts[i]- mean_f)/std_dev
            ftilda_users_posts.append(f_tilda)
            
    users_posts()
    filelist.extend(ftilda_users_posts)
   ################################################# 

#All the svector[] lists contain the S vector which is equation (5)  S D(i,t) = (FD(i,t+1) -FD i,t ) / Interval(Ei)

    svector_avg_microblogs_length = [(x - ftilda_avg_microblogs_length[i - 1])/interval_duration for i, x in enumerate(ftilda_avg_microblogs_length)][1:]
    svector_positive_negative_words = [(x - ftilda_positive_negative_words[i - 1])/interval_duration for i, x in enumerate(ftilda_positive_negative_words)][1:]
    svector_microblosg_url = [(x - ftilda_microblosg_url[i - 1])/interval_duration for i, x in enumerate(ftilda_microblosg_url)][1:]
    svector_microblogs_emoticons = [(x - ftilda_microblogs_emoticons[i - 1])/interval_duration for i, x in enumerate(ftilda_microblogs_emoticons)][1:]
    svector_positive_negaive_microblogs = [(x - ftilda_positive_negaive_microblogs[i - 1])/interval_duration for i, x in enumerate(ftilda_positive_negaive_microblogs)][1:]
    svector_first_person_pronouns = [(x - ftilda_first_person_pronouns[i - 1])/interval_duration for i, x in enumerate(ftilda_first_person_pronouns)][1:]
    svector_microblogs_with_hashtags = [(x - ftilda_microblogs_with_hashtags[i - 1])/interval_duration for i, x in enumerate(ftilda_microblogs_with_hashtags)][1:]
    svector_microblogs_with_at = [(x - ftilda_microblogs_with_at[i - 1])/interval_duration for i, x in enumerate(ftilda_microblogs_with_at)][1:]
    svector_microblogs_with_question = [(x - ftilda_microblogs_with_question[i - 1])/interval_duration for i, x in enumerate(ftilda_microblogs_with_question)][1:]
    svector_microblogs_with_exclamation = [(x - ftilda_microblogs_with_exclamation[i - 1])/interval_duration for i, x in enumerate(ftilda_microblogs_with_exclamation)][1:]
    svector_microblogs_with_question_exclamation = [(x - ftilda_microblogs_with_question_exclamation[i - 1])/interval_duration for i, x in enumerate(ftilda_microblogs_with_question_exclamation)][1:]
    svector_verified_users = [(x - ftilda_verified_users[i - 1])/interval_duration for i, x in enumerate(ftilda_verified_users)][1:]
    svector_users_friends = [(x - ftilda_users_friends[i - 1])/interval_duration for i, x in enumerate(ftilda_users_friends)][1:]
    svector_users_followers = [(x - ftilda_users_followers[i - 1])/interval_duration for i, x in enumerate(ftilda_users_followers)][1:]
    svector_users_posts = [(x - ftilda_users_posts[i - 1])/interval_duration for i, x in enumerate(ftilda_users_posts)][1:]
    
    filelist.extend(svector_avg_microblogs_length)
    filelist.extend(svector_positive_negative_words)
    filelist.extend(svector_microblosg_url)
    filelist.extend(svector_microblogs_emoticons)
    filelist.extend(svector_positive_negaive_microblogs)
    filelist.extend(svector_first_person_pronouns)
    filelist.extend(svector_microblogs_with_hashtags)
    filelist.extend(svector_microblogs_with_at)
    filelist.extend(svector_microblogs_with_question)
    filelist.extend(svector_microblogs_with_exclamation)
    filelist.extend(svector_microblogs_with_question_exclamation)
    filelist.extend(svector_verified_users)
    filelist.extend(svector_users_friends)
    filelist.extend(svector_users_followers)
    filelist.extend(svector_users_posts)
    
    # the final value which we have to predict i.e. if a tweet is a rumor or not   
    rumorlabel =data['rumor_label']
    if rumorlabel is False:
        falselist=[0]
        filelist.extend(falselist)
        falselist.clear()
    elif rumorlabel is True:
        truelist= [1]
        filelist.extend(truelist)
        truelist.clear()

    
    dataset_list.append(filelist) # appending data to of each event to the main list

    #################### the main() function ##################################################
"""           
if __name__ == '__main__':
    average_length_posts_of_user()
    positive_negative_words()
    microblogs_with_url()
    microblogs_with_emoticons()
    positive_negaive_microblogs()
    first_person_pronouns()
    microblogs_with_hashtags()
    microblogs_with_at()
    microblogs_with_question()
    microblogs_with_exclamation()
    microblogs_with_question_exclamation()
    verified_users()
    users_friends()
    users_followers()
    users_posts()
"""    

datasetfinal = np.array(dataset_list)

import random
random.shuffle(datasetfinal)# to shuffle our data

###********* TO write the data in a CSV file *********** we do this so as to avoid the time extracting the data from the original files(JSON files here)
#np.savetxt("rumor_detection.csv", datasetfinal, delimiter=",")
# open the csv file manually, insert a row on the top and  add something like 'xyz' on the first row first coloumn because pd.read_csv consider the first row as heading and ignores it
#datafile = pd.read_csv("rumor_detection.csv")
#X, y = datafile.iloc[:, :-1], datafile.iloc[:, -1]
##**********************************################

X, y = datasetfinal[:, :-1], datasetfinal[:, -1]

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X  = imputer.fit_transform(X)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(X)
print(kf)

# classifying the data as train and test
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# if not k-fold
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

# implementing linear SVM to our dataset
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train) # training our train dataset

# predicting the values on the test dataset
y_pred =classifier.predict(X_test)


# printing out the results to check the accuracy of the model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error,classification_report, confusion_matrix
from sklearn.cross_validation import cross_val_score

print('Test Accuracy: ',accuracy_score(y_test, y_pred) )
print('Test Precision: ',precision_score(y_test, y_pred, average="macro"))
print('Recall: ',recall_score(y_test, y_pred, average="macro")) 
print('F1 Score: ',f1_score(y_test, y_pred, average="macro"))
scorevalues = cross_val_score(classifier, X_test, y_test, cv=5)
print ("R^2 = ", scorevalues.mean())
print("mean squared error = ",mean_squared_error(y_test, y_pred).mean())


#####*********** END *****************############