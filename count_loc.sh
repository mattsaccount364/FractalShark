#!/bin/bash

cloc $(git ls-files -- . ':!:*.png' ':!:*.jpg' ':!:*.zip')
