package com.marpies.ane.facedetection {

    import flash.geom.Point;
    import flash.geom.Rectangle;

    /**
     * Object representing detected face.
     */
    public class Face {

        /**
         * The value that a probability is set to if it was not computed.
         */

		private var mDebug:String = "none";
		private var mopenfacePoints:String = "none";
		private var mposePoints:String = "none";
		private var mgazePoints:String = "none";
		private var mauValues:String = "none";
		
        /**
         * @private
         */
        public function Face() {
        }

        /**
         * @private
         */
        internal static function fromJSON( json:Object ):Face {
            var face:Face = new Face();
			face.mDebug = String(json.debugMessage);
			face.mopenfacePoints = String(json.openfacePoints).substr(3).slice(0,-3).split('"').join("");
			face.mposePoints = String(json.posePoints).substr(5).slice(0,-4).split('"').join("").split("[").join("");
			face.mgazePoints = String(json.gazePoints).substr(5).slice(0,-4).split('"').join("").split("[").join("");
			face.mauValues = String(json.auValues).substr(3).slice(0,-3).split('"').join("");
            return face;
        }

		// gets debug messages
		
		public function get debugMessage():String {
            return mDebug;
        }
		
		public function get openfacePoints():String {
            return mopenfacePoints;
        }
		
		public function get posePoints():String {
            return mposePoints;
        }
		
		public function get gazePoints():String {
            return mgazePoints;
        }
		public function get auValues():String {
            return mauValues;
        }

    }

}
