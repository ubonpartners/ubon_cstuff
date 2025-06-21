#include "BYTETracker.h"
#include "yaml_stuff.h"
#include <fstream>

BYTETracker::BYTETracker(const char *yaml_config/*float new_track_thresh,
						 float track_high_thresh,
						 float track_low_thresh,
						 float match_thresh1,
						 float match_thresh2,
						 float match_threshu,
						 float match_remove_dup,
						 float max_time_lost*/)
{
	YAML::Node yaml_base=yaml_load(yaml_config);
	this->new_track_thresh = yaml_base["new_track_thresh"].as<float>();
	this->track_high_thresh =yaml_base["track_high_thresh"].as<float>();
	this->track_low_thresh = yaml_base["track_low_thresh"].as<float>();
	this->match_thresh1 = yaml_base["match_thresh1"].as<float>();
	this->match_thresh2 = yaml_base["match_thresh2"].as<float>();
	this->match_threshu = yaml_base["match_threshu"].as<float>();
	this->match_remove_dup=yaml_base["match_remove_dup"].as<float>();
	this->max_time_lost = yaml_base["max_time_lost"].as<float>();
	//printf("%f %f %f %f\n",new_track_thresh,track_high_thresh,track_low_thresh,match_thresh1);
	//printf("%f %f %f %f\n",match_thresh2,match_threshu,match_remove_dup,max_time_lost);
	//exit(1);
	frame_id = 0;
	last_time = 0;

}

BYTETracker::~BYTETracker()
{
}

detection_list_t *BYTETracker::update(detection_list_t *dets, double rtp_time)
{
	// rtp time is microseconds

	float time_now=rtp_time;
	float time_delta_seconds=rtp_time-last_time;
	last_time=rtp_time;
	if (time_delta_seconds<0 or time_delta_seconds>2 or this->frame_id==0) time_delta_seconds=0.2;

	this->frame_id++;
	vector<STrack> activated_stracks;
	vector<STrack> refind_stracks;
	vector<STrack> removed_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> detections;
	vector<STrack> detections_low;

	vector<STrack> detections_cp;
	vector<STrack> tracked_stracks_swap;
	vector<STrack> resa, resb;
	vector<STrack> output_stracks;

	vector<STrack*> unconfirmed;
	vector<STrack*> tracked_stracks;
	vector<STrack*> strack_pool;
	vector<STrack*> r_tracked_stracks;

	int numDets = dets->num_person_detections;
	for(int i=0; i<numDets; i++)
	{
		detection_t *det=dets->person_dets[i];
		float score = det->conf;
		if (score>=track_low_thresh)
		{
			vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = det->x0*640;
			tlbr_[1] = det->y0*640;
			tlbr_[2] = det->x1*640;
			tlbr_[3] = det->y1*640;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score);
			if (score >= track_high_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
		}
	}


	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < (int)this->tracked_stracks.size(); i++)
	{
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]);
	}


	////////////////// Step 2: First association, with IoU //////////////////
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	STrack::multi_predict(strack_pool, this->kalman_filter, time_delta_seconds);

	vector<vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	vector<vector<int> > matches;
	vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh1, matches, u_track, u_detection);

	for (int i = 0; i < (int)matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id, time_now);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, time_now, false);
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < (int)u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());

	for (int i = 0; i < (int)u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, match_thresh2, matches, u_track, u_detection);

	for (int i = 0; i < (int)matches.size(); i++)
	{
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id, time_now);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, time_now, false);
			refind_stracks.push_back(*track);
		}
	}

	for (int i = 0; i < (int)u_track.size(); i++)
	{
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, match_threshu, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < (int)matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id, time_now);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < (int)u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	for (int i = 0; i < (int)u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->new_track_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id, time_now);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < (int)this->lost_stracks.size(); i++)
	{
		//if (this->frame_id - this->lost_stracks[i].end_frame() > 8)
		if (time_now - this->lost_stracks[i].end_time() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}

	for (int i = 0; i < (int)this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < (int)lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	for (int i = 0; i < (int)removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
		this->removed_track_ids.insert(removed_stracks[i].track_id);
	}
	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);

	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks, this->match_remove_dup);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());

	for (int i = 0; i < (int)this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}

	detection_list_t *out_dets=detection_list_create(output_stracks.size());
	for (int i = 0; i < (int)output_stracks.size(); i++)
	{
        detection_t *det=detection_list_add_end(out_dets);
        float x=output_stracks[i].tlwh[0];
        float y=output_stracks[i].tlwh[1];
        float w=output_stracks[i].tlwh[2];
        float h=output_stracks[i].tlwh[3];
        det->x0=(x)/640.0;
        det->x1=(x+w)/640.0;
        det->y0=(y)/640.0;
        det->y1=(y+h)/640.0;
        det->cl=0;
        det->conf=1.0;
		det->track_id=output_stracks[i].track_id;
    }
	return out_dets;
}
