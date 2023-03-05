import 'dart:async';

import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import 'main.dart';

void main() => runApp(const NostalgiaPage());

class NostalgiaPage extends StatelessWidget {
  const NostalgiaPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Video Player Demo',
      home: NostalgiaPage(),
    );
  }
}

class NostalgiaPageScreen extends StatefulWidget {
  const NostalgiaPageScreen({super.key});

  @override
  State<NostalgiaPageScreen> createState() => _NostalgiaPageScreenState();
}

class _NostalgiaPageScreenState extends State<NostalgiaPageScreen> {

  late Future<void> _initializeVideoPlayerFuture;

  @override
  void initState() {
    super.initState();

    // Create and store the VideoPlayerController. The VideoPlayerController
    // offers several different constructors to play videos from assets, files,
    // or the internet.

    // _controller = VideoPlayerController.network(
    //   'https://flutter.github.io/assets-for-api-docs/assets/videos/butterfly.mp4',
    // );

    // Initialize the controller and store the Future for later use.
    _initializeVideoPlayerFuture = controller.initialize();

    // Use the controller to loop the video.
    controller.setLooping(true);
  }

  @override
  void dispose() {
    // Ensure disposing of the VideoPlayerController to free up resources.
    controller.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // Use a FutureBuilder to display a loading spinner while waiting for the
      // VideoPlayerController to finish initializing.
      body: FutureBuilder(
        future: _initializeVideoPlayerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            // If the VideoPlayerController has finished initialization, use
            // the data it provides to limit the aspect ratio of the video.
            return AspectRatio(
              aspectRatio: controller.value.aspectRatio,
              // Use the VideoPlayer widget to display the video.
              child: NostalgiaPage(),
            );
          } else {
            // If the VideoPlayerController is still initializing, show a
            // loading spinner.
            return const Center(
              child: CircularProgressIndicator(),
            );
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // Wrap the play or pause in a call to `setState`. This ensures the
          // correct icon is shown.
          setState(() {
            // If the video is playing, pause it.
            if (controller.value.isPlaying) {
              controller.pause();
            } else {
              // If the video is paused, play it.
              controller.play();
            }
          });
        },
        // Display the correct icon depending on the state of the player.
        child: Icon(
          controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
        ),
      ),
    );
  }
}