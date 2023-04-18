import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'

const catsDir = './training/cat_or_dog/cat';
const dogsDir ='./training/cat_or_dog/dog';
const catsFiles = fs.readdirSync(catsDir);
const dogsFiles = fs.readdirSync(dogsDir);
const cats = catsFiles.map((file) => {
	const filePath = `${catsDir}/${file}`;
	const buffer = fs.readFileSync(filePath);
	const decodedImage = tf.node.decodeImage(buffer);
	const resizedImage = tf.image.resizeBilinear(decodedImage, [IMAGE_WIDTH, IMAGE_HEIGHT]);
	return resizedImage;
});

const dogs = dogsFiles.map((file) => {
	const filePath = `${dogsDir}/${file}`;
	const buffer = fs.readFileSync(filePath);
	const decodedImage = tf.node.decodeImage(buffer);
	const resizedImage = tf.image.resizeBilinear(decodedImage, [IMAGE_WIDTH, IMAGE_HEIGHT]);
	return resizedImage;
});

const images = cats.concat(dogs);
const labels = tf.tensor2d(
	Array.from({ length: cats.length }).fill([1, 0]).concat(Array.from({ length: dogs.length }).fill([0, 1]))
);

const model = new tf.Sequential({layers: [
		tf.layers.conv2d({filters: 16, kernelSize: [3,3], activation: 'relu', inputShape: [200,200,3]}),
		tf.layers.maxPool2d({poolSize: [2,2]}),
		tf.layers.conv2d({filters: 32, kernelSize: [3,3], activation: 'relu'}),
		tf.layers.maxPool2d({poolSize: [3,3]}),
		tf.layers.conv2d({filters: 64, kernelSize: [3,3], activation: 'relu'}),
		tf.layers.maxPool2d({poolSize: [2,2]}),
		tf.layers.dense({units: 512, activation: 'relu'}),
		tf.layers.dense({units: 1, activation: 'sigmoid'}),
	], name: 'animals'})
model.compile({optimizer: new tf.train.adam, loss: tf.losses.sigmoidCrossEntropy, metrics: [ tf.metrics.binaryAccuracy ]})
await model.fit(images, labels, {
	epochs: 100,
	shuffle: true
})