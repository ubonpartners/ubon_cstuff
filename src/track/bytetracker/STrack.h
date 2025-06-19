#pragma once

#include "kalmanFilter.h"

using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack(vector<float> tlwh_, float score);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter, float dt);
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	float end_time();

	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id, float time);
	void re_activate(STrack &new_track, int frame_id, float time, bool new_id = false);
	void update(STrack &new_track, int frame_id, float time);

public:
	bool is_activated;
	int track_id;
	int state;

	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;
	float time;
	float start_time;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;
	int _count;

private:
	byte_kalman::KalmanFilter kalman_filter;
};