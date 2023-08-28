toolbox_path = '/work/toolbox/matlab/vbmeg_svn_REV1590/external/yokogawa_lib';
addpath(toolbox_path);

sbj = 'sbj01';
sbj_dir = sprintf('/storage/dataset/MEG/internal/AnnotatedMovie_v1/tmp/meg/%s', sbj);
confile_path_info  = dir(sprintf('%s/%s', sbj_dir, '*/thirdParty/*.con'));

% 高速デジタル=162ch
% 低速デジタル=163ch
% 高速光=161ch
% 低速光=168ch

for i = 1:length(confile_path_info)
    fast_trigger_label = [];

    if ismember(i, [39,40]) &&  strcmp(sbj, 'sbj01')
        %sbj01のこのidの物だけ、triggerが入っていない
        fast_trig_id = 161;
        slow_trig_id = 168;
        trigger1_id = 3; % 低速光トリガーがvideo_id情報を伝える連続信号の最初のindex
        delay = 0; 
        % threshold=0.5;
        if i == 39 
            fast_trigger_label = [fast_trigger_label, [-1, -1]];
            max_interval=100; %
        elseif i == 40
            fast_trigger_label = [fast_trigger_label, [-1, -1]];
            % この場合だけ、video_idが10のところを11だとしてしまうので、10に手動で直す
        end

    else
        fast_trig_id = 162;
        slow_trig_id = 163;
        trigger1_id = 1;% 低速デジタルトリガーがvideo_id情報を伝える連続信号の最初のindex
        delay = 52;
        % threshold= 0;
        % continue
    end
    fast_ideal_value = 3;
    th_cnt = 20;
    if i < 39
        continue
    end
    disp(i)
    confile = fullfile(confile_path_info(i).folder, confile_path_info(i).name);
    data = getYkgwData(confile, 1, Inf); 
    data_size = size(data);
    slow_triggers = find(diff(data([slow_trig_id],:)) > 1); % list of [frame]
    slow_trig1 = slow_triggers(trigger1_id); % [frame]
    slow_trig2 = slow_triggers(trigger1_id+1); % [frame]
    slow_trig3 = slow_triggers(trigger1_id+2); % [frame]
    % fast_triggers = find(diff(data([fast_trig_id],:)) > 1); % list of [frame]  このthresholdでもうまく機能した
    fast_triggers = find(diff(data([fast_trig_id],:)>fast_ideal_value) > 0); % list of [frame]
    video_id = sum(fast_triggers > slow_trig1) - sum(fast_triggers > slow_trig2);
    part_id = sum(fast_triggers > slow_trig2) - sum(fast_triggers > slow_trig3);

    % tmp_fast_triggers = find(diff(data([fast_trig_id],:)) > threshold); % list of [frame]
    % fast_triggers = find(diff(data([fast_trig_id],:)>fast_ideal_value) > 0); % list of [frame]

    dummy_video_id = sum(fast_triggers > slow_trig1) - sum(fast_triggers > slow_trig2);
    dummy_part_id = sum(fast_triggers > slow_trig2) - sum(fast_triggers > slow_trig3);
    fast_trigger_label = [fast_trigger_label, repmat([-1],1,dummy_video_id)];
    fast_trigger_label = [fast_trigger_label, repmat([-1],1,dummy_part_id)];
    if part_id == 1
        trailor_frame = 0
    else
        trailor_frame = 300 % これによってMEG側が20,000フレームも移動するのおかしくない？ -> 正しい.2frameに一回triggerは発生する(1frameで立ち上がり、1frameで落ちるを繰り返す)% 。
    end
    fast_trigger_label = [fast_trigger_label, repmat([2],1,trailor_frame)];
    
    drama_start_trigger_id = find(fast_triggers > slow_trig3) + trailor_frame; % [index] trailor 300frame
    drama_start_trigger_id=drama_start_trigger_id(1);
    drama_end_triggeer_id = length(fast_triggers)-10; % [index] videoの最後に10 frameトリガー付き黒画面が出される
    video_frame_timing_trigger = fast_triggers(drama_start_trigger_id :drama_end_triggeer_id); %fast_triggers(drama_start_trigger_id:end);
    % fast_trigger_label = [fast_trigger_label, repmat([3], 1, drama_end_triggeer_id-drama_start_trigger_id+1)];
    cnt = 0;
    for i = drama_start_trigger_id:drama_end_triggeer_id
        cnt = cnt+1;
        label = 3;
        fast_trigger_label = [fast_trigger_label, label];
    end
    sprintf('count: %d', cnt) % 11975
    fast_trigger_label = [fast_trigger_label, repmat([4],1,10)];
    extra_length_start = 25000 % trailorの先頭が見えるはず
    extra_length_end = 1000;
    sprintf('%d, %d', video_frame_timing_trigger(1), video_frame_timing_trigger(end))
    % plot(data([fast_trig_id,slow_trig_id],video_frame_timing_trigger(1)-extra_length_start:video_frame_timing_trigger(1)+extra_length_start).'); saveas(gcf,sprintf('tmps/video_onsets/%dth_start.png', i));
    % plot(data([fast_trig_id,slow_trig_id],video_frame_timing_trigger(end)-extra_length_end:video_frame_timing_trigger(end)+extra_length_end).'); saveas(gcf,sprintf('tmps/video_onsets/%dth_end.png', i));
    sprintf('duration: %d frames (%f sec)', length(video_frame_timing_trigger), length(video_frame_timing_trigger) / 29.97 * 2)
    savefilename = sprintf('./triggers/%s/onset_trigger_video_%d-part_%d.csv', sbj, video_id, part_id)
    delay_vector = repmat([delay], 1, length(fast_triggers));
    sprintf('length: %d, %d, %d', length(fast_triggers), length(fast_trigger_label), length(delay_vector))
    labeled_trigger = cat(1, fast_triggers, fast_trigger_label, delay_vector);
    %break
    csvwrite(savefilename, labeled_trigger.');
    %if i >1
    %    break
    %end
end