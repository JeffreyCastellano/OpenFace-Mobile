#!/bin/sh
# change to the proper directory
pushd ~/Desktop/Test
# package the IPA - skip if already built
adt -package -target ipa-debug-interpreter -provisioning-profile Empath_Profile.mobileprovision -storetype pkcs12 -keystore certificate.p12 -storepass 9iptqz1 VisionTest.ipa Test-app.xml Test.swf -extdir ext -platformsdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS9.3.sdk/
# extract the IPA
cp VisionTest.ipa VisionTest.ipa.zip
unzip -o VisionTest.ipa.zip
rm VisionTest.ipa.zip
# copy the contents of the IPA to the location xcode wants
cp -r Payload/VisionTest.app/*       "${CONFIGURATION_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/"
                             cp -r VisionTest.app.dSYM "${CONFIGURATION_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/"
                             # remove the following files and folders to avoid signature errors when installing the app
                             rm "${CONFIGURATION_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/_CodeSignature/CodeResources"
                             rmdir "${CONFIGURATION_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/_CodeSignature"
                             rm "${CONFIGURATION_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/CodeResources"
                             rm "${CONFIGURATION_BUILD_DIR}/${CONTENTS_FOLDER_PATH}/PkgInfo"
                             # restore working directory
                             popd
