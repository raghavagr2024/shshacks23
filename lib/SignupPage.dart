import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

TextEditingController email = TextEditingController();
TextEditingController password = TextEditingController();

class SignupPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          const SizedBox(
            height: 70,
          ),

          EmailTextField(),

          SizedBox(height: 70,),
           PasswordTextField(),
          SizedBox(height: 70,),

         SignupButton()
        ],
      ),
    );
  }
}

class EmailTextField extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    // TODO: implement createState
    return _EmailTextField();
  }
}

class _EmailTextField extends State<EmailTextField> {
  @override
  Widget build(BuildContext context) {
    return Center (
        child: SizedBox(
          width: 250,
          child: TextFormField(
            controller: email,
            decoration: const InputDecoration(
              labelText: 'Email',

              focusedBorder: UnderlineInputBorder(

              ),
            ),
          ),
        )
    );
  }
}

class PasswordTextField extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    // TODO: implement createState
    return _PasswordTextField();
  }
}

class _PasswordTextField extends State<PasswordTextField> {
  @override
  Widget build(BuildContext context) {
    return Center (
        child: SizedBox(
          width: 250,
          child: TextFormField(
            controller: password,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'Password',

              focusedBorder: UnderlineInputBorder(

              ),
            ),
          ),
        )
    );
  }
}

class SignupButton extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return TextButton(onPressed: () {signupWithEmailAndPassword();}, child: Text("Sign up"));
  }

  Future<void> signupWithEmailAndPassword() async {
    try {
      final credential = await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: email.text,
        password: password.text,
      );
    } on FirebaseAuthException catch (e) {
      if (e.code == 'weak-password') {
        print('The password provided is too weak.');
      } else if (e.code == 'email-already-in-use') {
        print('The account already exists for that email.');
      }
    } catch (e) {
      print(e);
    }
    print("user created");
  }
}