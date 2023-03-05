const { initializeApp } = require("firebase/app");
const { getStorage, ref, getBytes, listAll } = require("firebase/storage");
const { appendFile } = require('fs')
const fs = require('fs');

const uid = process.argv[2]
console.log('uuid' + uid)

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

const firebaseConfig = {
  apiKey: "AIzaSyBG-Q2sScBxXFvl3TH-XsRHpqeUPfXzODA",
  authDomain: "wellness-shshacks.firebaseapp.com",
  projectId: "wellness-shshacks",
  storageBucket: "wellness-shshacks.appspot.com",
  messagingSenderId: "39579069719",
  appId: "1:39579069719:web:31faf62df0cf53156b9b52",
  measurementId: "G-TJZWQ9FJSV"
};

const app = initializeApp(firebaseConfig);
const storage = getStorage(app);


const pathRef = ref(storage, uid);
listAll(pathRef).then((res) => {
  let newRef = ref(storage, [...res.items].sort().reverse()[0].toString())
  getBytes(newRef).then( bytes => appendFile('videos/' + uid + '.mp4', Buffer.from(bytes),
    err => {
      if (err) {
        console.log(err);
      };
    }
  ));
})

//const gsReference = ref(storage, 'gs://wellness-shshacks.appspot.com/testuser/testrecording.mp3');

console.log('done')