#pragma once
#include <unordered_set>
#include <iostream>
#include "detections.h"
#include "STrack.h"

class BYTETracker
{
public:
	BYTETracker(const  char *yaml_config);
	~BYTETracker();

	// vector<STrack> update(int* num_dets, float* Boxes, float* Scores, int* Classes, int batch_num, int max_det_per_batch);
	detections_t *update(detections_t *dets, double rtp_time);

	vector<STrack> removed_stracks;
	std::unordered_set<int> removed_track_ids;



private:
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb, float match_remove_dup);

	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
	float new_track_thresh;
	float track_high_thresh;
	float track_low_thresh;
	float match_thresh1;
	float match_thresh2;
	float match_threshu;
	float match_remove_dup;
	float max_time_lost;

	int frame_id;
	double last_time;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};
