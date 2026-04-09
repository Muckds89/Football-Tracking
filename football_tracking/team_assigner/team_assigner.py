from sklearn.cluster import KMeans
import cv2
import numpy as np
from collections import defaultdict, Counter


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id
    
    def get_torso_crop(self,frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        h = y2 - y1
        w = x2 - x1

        if h <= 0 or w <= 0:
            return None

        # middle torso region
        tx1 = x1 + int(0.2 * w)
        tx2 = x1 + int(0.8 * w)
        ty1 = y1 + int(0.15 * h)
        ty2 = y1 + int(0.55 * h)

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        return crop


    def classify_player_team(self,frame, bbox, vest_ratio_thresh=0.08):
        crop = self.get_torso_crop(frame, bbox)
        if crop is None:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([18, 70, 70])
        upper_yellow = np.array([40, 255, 255])

        lower_orange = np.array([8, 80, 80])
        upper_orange = np.array([20, 255, 255])

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        vest_mask = cv2.bitwise_or(yellow_mask, orange_mask)

        vest_ratio = np.count_nonzero(vest_mask) / vest_mask.size

        if vest_ratio >= vest_ratio_thresh:
            team = "vest_team"
        else:
            team = "other_team"
        return team, vest_ratio       
     
    def assign_player_teams_from_video(self,video_path, tracks):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_num = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num >= len(tracks["players"]):
                    break

                player_tracks = tracks["players"][frame_num]

                for player_id, info in player_tracks.items():
                    bbox = info["bbox"]

                    team, vest_ratio = self.classify_player_team(frame, bbox)
                    tracks["players"][frame_num][player_id]["team"] = team
                    tracks["players"][frame_num][player_id]["vest_ratio"] = float(vest_ratio)

                frame_num += 1

        finally:
            cap.release()

        return tracks        
    
    def smooth_player_teams(self,tracks):
        votes = defaultdict(list)

        for frame_players in tracks["players"]:
            for player_id, info in frame_players.items():
                team = info.get("team")
                if team is not None and team != "unknown":
                    votes[player_id].append(team)

        stable_team = {}
        for player_id, team_votes in votes.items():
            if len(team_votes) == 0:
                stable_team[player_id] = "unknown"
            else:
                stable_team[player_id] = Counter(team_votes).most_common(1)[0][0]

        for frame_players in tracks["players"]:
            for player_id, info in frame_players.items():
                info["team"] = stable_team.get(player_id, "unknown")

        return tracks
