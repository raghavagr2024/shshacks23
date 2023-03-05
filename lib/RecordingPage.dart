import 'dart:io';

import 'package:better_open_file/better_open_file.dart';
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'widgets/custom_media_preview.dart';

void main() {
  runApp(const RecordingPage());
}

class RecordingPage extends StatelessWidget {
  const RecordingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'camerAwesome',
      home: CameraPage(),
    );
  }
}

class CameraPage extends StatelessWidget {
  const CameraPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CameraAwesomeBuilder.custom(
        sensor: Sensors.front,
        builder: (cameraState, previewSize, previewRect) {
          return cameraState.when(
            onPreparingCamera: (state) =>
                const Center(child: CircularProgressIndicator()),
            onVideoMode: (state) => RecordVideoUI(state, recording: false),
            onVideoRecordingMode: (state) =>
                RecordVideoUI(state, recording: true),
          );
        },
        saveConfig: SaveConfig.video(
          pathBuilder: () async {
            return "some/path.mp4";
          },
        ),
      ),
    );
  }
/*
 *       body: CameraAwesomeBuilder.awesome(
        sensor: Sensors.front,
        saveConfig: SaveConfig.video(
          pathBuilder: () => _path(CaptureMode.video)
        ),
        flashMode: FlashMode.auto,
        aspectRatio: CameraAspectRatios.ratio_16_9,
        previewFit: CameraPreviewFit.fitWidth,
        onMediaTap: (mediaCapture) {
          OpenFile.open(mediaCapture.filePath);
        },
      ),
 */

  Future<String> _path(CaptureMode captureMode) async {
    final Directory extDir = await getTemporaryDirectory();
    final testDir =
        await Directory('${extDir.path}/test').create(recursive: true);
    final String fileExtension =
        captureMode == CaptureMode.photo ? 'jpg' : 'mp4';
    final String filePath =
        '${testDir.path}/${DateTime.now().millisecondsSinceEpoch}.$fileExtension';
    return filePath;
  }
}

class RecordVideoUI extends StatelessWidget {
  final CameraState state;
  final bool recording;

  const RecordVideoUI(this.state, {super.key, required this.recording});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Spacer(),
        Padding(
          padding: const EdgeInsets.only(bottom: 20, left: 20, right: 20),
          child: Row(
            children: [
              AwesomeCaptureButton(state: state),
              const Spacer(),
              
            ],
          ),
        ),
      ],
    );
  }
}

class Timer extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    // TODO: implement createState
    return _Timer();
  }
}

class _Timer extends State<Timer> {
  int time = 60;
  @override
  Widget build(BuildContext context) {
    return Text(formatTime());
  }
  String formatTime() {
    if (time == 60) {
      return "1:00";
    } else {
      return "0:$time";
    }
  }
}