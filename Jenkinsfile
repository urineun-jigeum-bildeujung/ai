// ai 레포 CI/CD — 2026-09-21 첫 작성 (인프라), 2026-09-23 nutrition 추가,
// 2026-09-23 deployReady 게이트 + repurchase 추가.
//
// SERVICES 목록에 있는 서비스만 감지/빌드 대상. repurchase는 GHCR 자체 CI/CD를
// ECR로 전환하기로 AI팀과 합의(#117) — GHCR 게시 자동화는 AI팀이 별도로 끔.
//
// deployReady: false인 서비스는 빌드+Trivy 스캔까지만 하고 ECR push/GitOps 갱신은
// 건너뜀. recommendation은 ECR 레포/gitops-value 값파일이 아직 없어서, repurchase는
// 컨테이너 내부 명령이 아직 계약 검증용만 연결돼 있어서(AI팀 요청, #117) 둘 다
// false로 시작. 준비되면 해당 서비스의 값만 true로 바꾸는 PR 한 줄로 끝남.
def SERVICES = [
    [name: 'recommendation', path: 'recommendation/endtoend', dockerfile: 'recommendation/endtoend/Dockerfile', deployReady: false],
    [name: 'nutrition', path: 'nutrition', dockerfile: 'nutrition/Dockerfile', deployReady: true],
    [name: 'repurchase', path: 'repurchase/data_analysis', dockerfile: 'repurchase/data_analysis/Dockerfile', deployReady: false],
]

def changedServices = []
def deployableServices = []
def imageTag = ''
def isRealDeploy = false

pipeline {
    options {
        disableConcurrentBuilds()
    }

    // Jenkins가 K8s 파드로 떠서 도커 데몬이 없음 — kaniko가 daemon 없이 이미지를 빌드함.
    // sever Jenkinsfile과 동일한 kaniko/trivy/crane/awscli 구성을 그대로 재사용.
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: jenkins-kaniko
  containers:
    - name: kaniko
      image: gcr.io/kaniko-project/executor:debug
      command:
        - /busybox/cat
      tty: true
      resources:
        requests:
          cpu: 50m
          memory: 256Mi
          ephemeral-storage: 1Gi
        limits:
          cpu: "2"
          memory: 3Gi
          ephemeral-storage: 6Gi
    - name: trivy
      image: aquasec/trivy:0.74.0
      command:
        - sleep
      args:
        - 99d
      resources:
        requests:
          cpu: 30m
          memory: 128Mi
          ephemeral-storage: 512Mi
        limits:
          cpu: "1"
          memory: 1Gi
          ephemeral-storage: 6Gi
    - name: crane
      image: gcr.io/go-containerregistry/crane:debug
      command:
        - sleep
      args:
        - 99d
      resources:
        requests:
          cpu: 20m
          memory: 32Mi
          ephemeral-storage: 128Mi
        limits:
          cpu: 500m
          memory: 256Mi
          ephemeral-storage: 512Mi
    - name: awscli
      # kaniko는 ECR 인증이 내장돼 있지만 crane push는 그게 없어서 401로 실패함
      # (sever Jenkinsfile에서 2026-09-14 실제로 겪은 문제, 동일 원인으로 미리 반영).
      image: amazon/aws-cli:2.29.0
      command:
        - sleep
      args:
        - 99d
      resources:
        requests:
          cpu: 20m
          memory: 64Mi
          ephemeral-storage: 128Mi
        limits:
          cpu: 500m
          memory: 256Mi
          ephemeral-storage: 256Mi
      volumeMounts:
      - mountPath: "/home/jenkins/agent"
        name: "workspace-volume"
        readOnly: false
"""
        }
    }

    parameters {
        // 수동 빌드 시 여기 값 채워서 실행 = 특정 서비스만 강제로 빌드 (비워두면 자동 감지).
        string(name: 'SERVICE', defaultValue: '', description: '수동 빌드할 서비스명 (비워두면 변경분 자동 감지)')
    }

    environment {
        IMAGE_REGISTRY     = '297165773875.dkr.ecr.ap-northeast-2.amazonaws.com/petflow'
        GITOPS_VALUE_REPO  = 'https://github.com/urineun-jigeum-bildeujung/gitops-value.git'
    }

    stages {
        stage('Detect Deploy') {
            steps {
                script {
                    // sever와 동일한 기준: PR 검증 빌드나 develop 아닌 브랜치 push는
                    // 빌드+스캔까지만 하고 ECR push/GitOps 갱신은 안 함. 사람이 수동으로
                    // "Build Now" 누른 것도 실배포에서 제외(재실행이 실배포로 오인되는
                    // 사고 방지 — sever에서 2026-09-13 CodeRabbit 리뷰로 발견된 문제와 동일).
                    def isManualTrigger = !currentBuild.getBuildCauses('hudson.model.Cause$UserIdCause').isEmpty()
                    isRealDeploy = (env.CHANGE_ID == null) && (env.BRANCH_NAME == 'develop') && !isManualTrigger
                    imageTag = sh(script: 'git rev-parse HEAD', returnStdout: true).trim()

                    if (params.SERVICE?.trim()) {
                        changedServices = [params.SERVICE.trim()]
                    } else {
                        def baseSha = ''
                        if (env.CHANGE_ID) {
                            // PR 빌드: Declarative Checkout SCM이 PR 빌드에서 refs/pull/<N>/head만
                            // fetch하고 대상 브랜치는 로컬에 없어서, merge-base 전에 먼저 fetch해야 함
                            // (sever Jenkinsfile에서 2026-09-14 겪은 문제와 동일한 원인).
                            sh "git fetch --no-tags origin ${env.CHANGE_TARGET}:refs/remotes/origin/${env.CHANGE_TARGET}"
                            baseSha = sh(
                                script: "git merge-base HEAD origin/${env.CHANGE_TARGET}",
                                returnStdout: true
                            ).trim()
                        } else {
                            baseSha = env.GIT_PREVIOUS_SUCCESSFUL_COMMIT ?:
                                sh(script: 'git rev-parse HEAD~1', returnStdout: true).trim()
                        }

                        changedServices = SERVICES.findAll { svc ->
                            sh(script: "git diff --quiet ${baseSha} HEAD -- ${svc.path}", returnStatus: true) != 0
                        }.collect { it.name }
                    }

                    echo "변경된 서비스: ${changedServices}"
                    echo "실배포 여부: ${isRealDeploy}"
                }
            }
        }

        stage('Build & Scan') {
            when {
                expression { return !changedServices.isEmpty() }
            }
            steps {
                script {
                    changedServices.each { svcName ->
                        def svc = SERVICES.find { it.name == svcName }
                        def tarFile = "${svc.name}.tar"
                        def imageRef = "${env.IMAGE_REGISTRY}/${svc.name}:${imageTag}"

                        // 빌드만(push 안 함) -> Trivy CRITICAL 스캔(걸리면 실패) -> 실배포일
                        // 때만 push. 스캔 통과 못 한 이미지는 push 코드 경로를 안 타서
                        // 물리적으로 못 올라감 (sever와 동일한 안전장치).
                        container('kaniko') {
                            sh """
                                /kaniko/executor \\
                                  --context=`pwd`/${svc.path} \\
                                  --dockerfile=`pwd`/${svc.dockerfile} \\
                                  --destination=${imageRef} \\
                                  --no-push \\
                                  --tarPath=${tarFile} \\
                                  --cleanup
                            """
                        }

                        container('trivy') {
                            sh """
                                trivy image --input ${tarFile} \\
                                  --severity CRITICAL --exit-code 1 --ignore-unfixed
                            """
                        }

                        if (isRealDeploy && svc.deployReady) {
                            if (!env.getProperty('ECR_LOGGED_IN')) {
                                container('awscli') {
                                    sh "aws ecr get-login-password --region ap-northeast-2 > ecr-token.txt"
                                }
                                container('crane') {
                                    sh "crane auth login ${env.IMAGE_REGISTRY.split('/')[0]} --username AWS --password-stdin < ecr-token.txt"
                                }
                                sh "rm -f ecr-token.txt"
                                env.ECR_LOGGED_IN = 'true'
                            }
                            container('crane') {
                                sh "crane push ${tarFile} ${imageRef}"
                            }
                        }

                        sh "rm -f ${tarFile}"
                    }
                }
            }
        }

        stage('Update GitOps') {
            when {
                expression {
                    return isRealDeploy && changedServices.any { svcName ->
                        SERVICES.find { it.name == svcName }?.deployReady
                    }
                }
            }
            steps {
                script {
                    deployableServices = changedServices.findAll { svcName ->
                        SERVICES.find { it.name == svcName }.deployReady
                    }
                }

                withCredentials([usernamePassword(
                    credentialsId: 'gitops-value-push',
                    usernameVariable: 'GIT_USER',
                    passwordVariable: 'GIT_TOKEN'
                )]) {
                    // 토큰을 URL에 안 박고 git credential helper로 그 순간에만 넘겨서,
                    // clone한 저장소의 .git/config에 토큰이 평문으로 남지 않게 한다
                    // (CodeRabbit 리뷰 반영, 2026-09-23). 작은따옴표(''')로 감싸야
                    // $GIT_USER/$GIT_TOKEN이 Groovy가 아니라 쉘 실행 시점에 실제
                    // 환경변수로 치환된다 — 큰따옴표를 쓰면 Groovy가 스크립트 생성
                    // 시점에 값을 미리 텍스트로 박아버려서 이 대책이 무의미해진다.
                    // set +x로 명령어 자체가 로그에 찍히는 것도 같이 막는다.
                    sh '''
                        set +x
                        rm -rf gitops-value-checkout
                        git -c credential.helper='!f() { echo "username=$GIT_USER"; echo "password=$GIT_TOKEN"; }; f' \
                            clone https://github.com/urineun-jigeum-bildeujung/gitops-value.git gitops-value-checkout
                    '''
                }

                // yq로 .image.tag만 정확히 갱신 (sed는 YAML 구조를 몰라서 다른 tag: 줄까지
                // 잘못 건드릴 위험이 있음 — sever Jenkinsfile과 동일한 이유로 yq 사용).
                sh '''
                    curl -sL https://github.com/mikefarah/yq/releases/download/v4.44.3/yq_linux_amd64 -o /tmp/yq
                    chmod +x /tmp/yq
                '''

                script {
                    deployableServices.each { svcName ->
                        sh """
                            /tmp/yq -i '.image.tag = "${imageTag}"' gitops-value-checkout/values/dev/services/${svcName}/values.yaml
                        """
                    }
                }

                dir('gitops-value-checkout') {
                    // 커밋 메시지엔 비밀값이 없어서 그대로 Groovy 보간(""")을 써도 안전하다.
                    sh """
                        git config user.email 'jenkins@petflow.local'
                        git config user.name 'jenkins-ci'
                        git add values/
                        git diff --cached --quiet && echo '변경 없음, commit 생략' || git commit -m 'chore: deploy ${deployableServices.join(", ")} @ ${imageTag}'
                    """

                    withCredentials([usernamePassword(
                        credentialsId: 'gitops-value-push',
                        usernameVariable: 'GIT_USER',
                        passwordVariable: 'GIT_TOKEN'
                    )]) {
                        sh '''
                            set +x
                            git -c credential.helper='!f() { echo "username=$GIT_USER"; echo "password=$GIT_TOKEN"; }; f' push
                        '''
                    }
                }
            }
        }
    }
}
