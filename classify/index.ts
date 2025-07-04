import { Request, Response } from "express";
import * as tf from "@tensorflow/tfjs-node";
import * as mobilenet from "@tensorflow-models/mobilenet";

let model: mobilenet.MobileNet | null = null;
async function getModel() {
  if (!model) {
    model = await mobilenet.load({version: 2});
  }
  return model;
}

export const classifyDish = async (req: Request, res: Response) => {
  try {
    const imageData = req.body.image;

    if (!imageData) {
      return res.status(400).json({ error: "No image provided" });
    }

    console.log({imageData})

    // Ensure imageData is a Buffer
    let imageBuffer: Buffer;

    if (Buffer.isBuffer(imageData)) {
      imageBuffer = imageData;
    } else if (
      imageData &&
      imageData.type === "Buffer" &&
      Array.isArray(imageData.data)
    ) {
      imageBuffer = Buffer.from(imageData.data);
    } else if (typeof imageData === "string") {
      imageBuffer = Buffer.from(
        imageData.replace(/^data:image\/\w+;base64,/, ""),
        "base64"
      );
    } else {
      return res.status(400).json({ error: "Invalid image data" });
    }
    
    console.log({imageBuffer})

    const model = await getModel();
    const imageTensor = tf.node.decodeImage(imageBuffer) as tf.Tensor3D;
    const predictions = await model.classify(imageTensor, 5);
    imageTensor.dispose();

    const topPredictions = predictions.map((p) => ({
      dish: p.className,
      confidence: Math.round(p.probability * 100),
    }));

    return res.status(200).json({
      confidence: topPredictions[0]?.confidence || 0,
      predictions: topPredictions,
    });
  } catch (error) {
    console.error("Classify error:", error);
    return res.status(500).json({ error: "Server error" });
  }
};
