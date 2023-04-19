import * as tfNode from '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'

const catsDir = './training/cat_or_dog/cat';
const dogsDir ='./training/cat_or_dog/dog';
const catsFiles = fs.readdirSync(catsDir);
const dogsFiles = fs.readdirSync(dogsDir);
const cats = catsFiles.map((file) => {
	const filePath = `${catsDir}/${file}`;
	const buffer = fs.readFileSync(filePath);
	const decodedImage = tfNode.node.decodeImage(buffer);
	return tf.image.resizeBilinear(decodedImage, [ 200, 200 ]);
});
const dogs = dogsFiles.map((file) => {
	const filePath = `${dogsDir}/${file}`;
	const buffer = fs.readFileSync(filePath);
	const decodedImage = tfNode.node.decodeImage(buffer);
	return tf.image.resizeBilinear(decodedImage, [ 200, 200 ]);
});
const images = cats.concat(dogs);
const labels = tf.tensor2d(
	Array.from({ length: cats.length }).fill([1,0]).concat(Array.from({ length: dogs.length }).fill([0,1]))
);
console.log(labels.arraySync())
const model = tf.sequential()

model.add(tf.layers.conv2d({
	inputShape: [200, 200, 3],
	filters: 16,
	kernelSize: 3,
	activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({
	poolSize: 2,
	strides: 2
}));
model.add(tf.layers.conv2d({
	filters: 32,
	kernelSize: 3,
	activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({
	poolSize: 2,
	strides: 2
}));
model.add(tf.layers.conv2d({
	filters: 64,
	kernelSize: 3,
	activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({
	poolSize: 2,
	strides: 2
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
	units: 512,
	activation: 'relu'
}));
model.add(tf.layers.dense({
	units: 2,
	activation: 'sigmoid'
}));

model.compile({
	optimizer: 'adam',
	loss: tf.losses.sigmoidCrossEntropy,
	metrics: ['accuracy']
});
await model.fit(tf.stack(images), labels, {
	batchSize: 32,
	epochs: 5,
	validationData: [tf.stack(dogs),tf.tensor2d(Array.from({ length: dogs.length }).fill([0,1]))]
	
})
console.log(await model.predict(tf.stack(dogs)).arraySync());
//console.log(await model.predict(tf.stack(dogs)).data());
//console.log(await model.predict(tf.stack(dogs)).data());
console.log(await model.predict(tf.stack(cats)).arraySync());